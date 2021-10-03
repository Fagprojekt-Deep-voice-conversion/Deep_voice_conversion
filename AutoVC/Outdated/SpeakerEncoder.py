from builtins import int

import numpy as np
import torch
from torch import nn
from ge2e import GE2ELoss
from Kode.Preprocessing_WAV import Preproccesing

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        self.hidden_size = 768
        self.num_layers = 2
        n_mels = 80
        
        self.lstm = nn.LSTM(input_size = n_mels, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.linear = nn.Linear(in_features = 768, out_features = 256)
        self.relu = nn.ReLU()


    def forward(self, mels: torch.FloatTensor):
        hidden, _ = self.lstm(mels)
        raw_embedding = self.relu(self.linear(hidden[-1]))
        embedding = raw_embedding/torch.norm(raw_embedding, dim = 1, keepdim = True)
        return embedding


    def train(self, X, y, loss = "GE2E", optimiser = 'SGD'):
        criterion = GE2ELoss(init_w = 10, init_b = -5, loss_method = 'softmax')
        optimiser = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        optimiser.zero_grad()

        output = self.forward(X)
        loss = criterion(output, y)
        loss.backward()
        optimiser.step()


    def PartialSlicer(self, wav):
        Prep = Preproccesing()
        partial_n_frames = 1.6 # seconds
        sr = Prep.sampling_rate
        fft_len = Prep.n_fft
        n_frames = 2
        frame_step = int(partial_n_frames / (fft_len / sr)) # = 25 frames :-)
        shift = int(frame_step/2) # 50 % of framestep

        mels = Prep.spec_Mel(wav, ["Hej", "dig"])



        for label in mels:
            for mel in label:
                mel_slices = []
                print(np.shape(mel))






        print(frame_step)


