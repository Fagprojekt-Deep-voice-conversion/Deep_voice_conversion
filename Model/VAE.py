import torch
import os
import sys
path = sys.path[0]
os.chdir(path)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))

