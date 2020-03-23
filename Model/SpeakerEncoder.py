import numpy as np
import torch
from torch import nn


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

