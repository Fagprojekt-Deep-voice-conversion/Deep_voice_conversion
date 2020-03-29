import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """Encoder module:
    AutoVC paper: Figure 3(a)
    The encoder takes a mel spectrogram combined with the speaker embedding
    and takes these concatenated features and inserts it into three 5X1 convolutional
    layers, followed by batch normalisation and ReLU activation. The output of these
    layers are then feed into two bidirectional LSTM layers.
    to create the bottleneck, the forward and backwards output er downsampled by 32.
    the downsampling is performed differently for the forward and backward output and is
    visualised in figure 3(e) and (f).
    ------------------------------------------------------
    Parameters:
    input: 80-dimensional mel spectrogram
    Number of ConvNorm layers: 3
        - Number of channels: 512
    Number of BLSTM layers: 2
        - combined dimension: 64

    ------------------------------------------------------
    Output:
    Content embedding presented as 2 32-by-T/32 matrices, denoted as C1-> and C1<- in the paper.
    
    """

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 + dim_emb if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes
