"""
AutoVC model from https://github.com/auspicious3000/autovc. See LICENSE.txt

The model has the same architecture as proposed in "AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss" - referred to as 'the Paper'
Everything is shown in figure 3 in the Paper - please have this by hand when reading through
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    """
    Creates a linear layer.
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        """
        The linear layer with in dimension = in_dim  and out dimension = out_dim
        Weights are initialised using the uniform Xavier function
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """
    Creates a convolutional layer (with normalisation ??)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        """
        The convolutional layer with in dimension = in_channels and out dimension = out_channels
        Kernel size is default 1.
        Weights are initialised using the Uniform Xavier function
        """
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """
    Content Encoder module as presented in the Paper Figure 3.a

    First the signal is processed through 3 convolutional layers of in_dim = out_dim = 512
    The kernel size is of dim 5x5 and zero padding is performed by padding 2 times.

    After the convolutions the signal is processed through 2 Bidirectional LSTM layers with bottleneck dimension 32
    The BLSTM finally produces a forward output and a backward output of dimension 32 for each timestep (frame) of the input

    The forward input is then downsampled by taking the output at time (31, 63, 95, ...)
    The backward input is then downsampled by taking the output at time (0, 32, 64, ...)

    Notice that the downsampling here is inconsistent in what stated in the Paper... In the Paper it is opposite downsampling for backward and forward
    """
    def __init__(self, dim_neck, dim_emb, freq):
        """
        params:
        dim_neck: Dimension of the bottleneck - set to 32 in the paper
        dim_emb: Dimension of the speaker embedding - set to 256 in the paper
        freq: sampling frequency for downsampling - set to 32 in the paper
        """
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        """ The 3 convolutional layers"""
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        """ The 2 BLSTM layers """
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        """
        Process spectrogram of dim (batch_size, time_frames, n_mels) and the speaker embedding of dim 256
        n_mels is set to be 80.

        params:
        x: mel spectrogram
        c_org: embedding of speaker
        """

        """ Concatenates Spectrogram and speaker embedding"""
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        """ Process through convolutional layers with ReLu activation function """
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        """ Process through BLSTM layers and obtain forward and backward output"""
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        """
        Donwsampling...
        These lines are from the git repo  https://github.com/auspicious3000/autovc but only functional for certain inputs...
        """
        #codes = []
        #for i in range(0, outputs.size(1), self.freq):
        #    print(i)
        #    codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))


        """
        Downsampling...
        Slight Adjustments to be consistent with the paper
        Down sampling as in figure 3, box e and f
        """
        codesA = [out_forward[:, i, :] for i in range(self.freq-1, outputs.size(1), self.freq)]  # (31, 63, 95, ...)
        codesB = [out_backward[:, i, :] for i in range(0, outputs.size(1), self.freq)]  # (0, 32, 64, ... )

        """ Return a list of containing C1 -> and C1 <- as in figure 3"""
        codes = [codesA, codesB]

        return codes
      
        
class Decoder(nn.Module):
    """
    Decoder module as proposed in the Paper figure 3.c
    The Decoder takes a input of the upsampled encoder outputs concatenated with the speaker embedding
    Input dim = 32 * 2 + 256 = 320

    First the input is process through a single LSTM layer (inconsistent with paper!)

    Secondly the signal is processed through 3 convolutional layers of in_dim = out_dim = 512
    The kernel size is of dim 5x5 and zero padding is performed by padding 2 times.
        (as in the encoder)

    Afterwards the signal i processed through 2 LSTM layers with out dimension 1024 (in the paper it's 3 layers...)

    Finally a linear projection to dim 80 is performed.
    The output has the same dimensions as the mel input (batch_size, time_frames, n_mels (80) )

    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        """
        params:
        dim_neck: the bottleneck dimension (set to 32 in the paper)
        dim_emb: speaker embedding dimension (set to 256 in the paper)
        dim_pre: out dimension of the pre LSTM layer (set to 512 in paper)
        """
        super(Decoder, self).__init__()

        """ The pre-LSTM layer. In: 320, out: 512 """
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)

        """ 3 convolutional layers with batch Normalisation """
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        """ Secondary double LSTM layer """
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        """ Final linear projection to original dimension """
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        """
        params:
        x: dimension 320. C1 <- + C1 -> + Speaker embedding
        """
        """ Sends signal through the first LSTM layer """
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        """ Sends signal through the convolutional layers with batch normalisation and ReLu activation """
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        """ Through the secondary double LSTM layer """
        outputs, _ = self.lstm2(x)

        """ Final projection unto original dimension """
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """
    Postnet: the last part of figure 3.c in the Paper
        - Five 1-d convolution with 512 channels and kernel size 5
        - 1 layer with in dim 80 and out dim 512
        - 3 layers with in dim = out dim = 512
        - 1 layer with in dim 512 and out dim 80
        - all with tanh activaion funcion
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        """ The first layer """
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )
        """ The 3 midder layers """
        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )
        """ The Final Layer """
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        """ Takes the output from the decoder module as input """

        """ Through the first 4 layers """
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        """ 
        Out through the 5th and final layer 
            - Out dimension equal to original spectrogram input
        """
        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """
    Generator network. The entire thing pieced together (figure 3a and 3c)
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        """
        params:
        dim_neck: dimension of bottleneck (set to 32 in the paper)
        dim_emb: dimension of speaker embedding (set to 256 in the paper)
        dim_pre: dimension of the input to the decoder (output of first LSTM layer) (set to 512 in the paper)
        """
        super(Generator, self).__init__()
        self.freq = freq
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        """
        params:
        x: spectrogram batch dim: (batch_size, time_frames, n_mels (80) )
        c_org: Speaker embedding of source dim (batch size, 256)
        c_trg: Speaker embedding of target (batch size, 256)
        """

        """ Pass x and c_org through encoder and obtain downsampled C1 -> and C1 <- as in figure 3"""
        codes = self.encoder(x, c_org)

        """ 
        If no target provide output the content codes from the content encoder 
        This is for the loss function to easily produce content codes from the final output of AutoVC
        """
        if c_trg is None:
            content_codes = torch.cat([torch.cat(code, dim=-1) for code in codes], dim=-1)
            return content_codes

        """ 
        Upsampling as in figure 3e-f.
        Recall the forward output of the decoder is downsampled at time (31, 63, 95, ...) and the backward output at (0, 32, 64, ...)
        The upsampling copies the downsampled to match the original input.
        E.g input of dim 100:
            Downsampling
            - Forward: (31, 63, 95)
            - Backward: (0, 32, 64, 96)
            Upsampling:
            - Forward: (0-31 = 31, 32-63 = 63, 64-100 = 95)
            - Backward: (0-31 = 0, 32-63 = 32, 64-95 = 64, 96-100 = 96)
        """
        tmp = []
        for code in codes:
            L = len(code)
            diff = x.size(1) - L * self.freq
            Up_Sampling = [sample.unsqueeze(1).expand(-1, self.freq + bool( (i+1) == L) * diff, -1) for i, sample in enumerate(code)]
            tmp.append(torch.cat(Up_Sampling, dim = 1))

        """ Concatenates upsampled content codes with target embedding. Dim = (batch_size, input time frames, 320) """
        code_exp = torch.cat(tmp, dim=-1)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)

        """ Sends concatenate encoder outputs through the decoder """
        mel_outputs = self.decoder(encoder_outputs)


        """ Sends the decoder outputs through the 5 layer postnet and adds this output with decoder output for stabilisation (section 4.3)"""
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)

        """ 
        Prepares final output
        mel_outputs: decoder outputs
        mel_outputs_postnet: decoder outputs + postnet outputs
        contetn_codes: the codes from the content encoder
        """
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        content_codes = torch.cat([torch.cat(code, dim = -1) for code in codes], dim = -1)

        return mel_outputs, mel_outputs_postnet, content_codes




