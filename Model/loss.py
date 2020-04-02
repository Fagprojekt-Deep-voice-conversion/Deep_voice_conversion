import os, sys
import numpy as np
import torch
from Model.AutoVC.autovc_master.model_vc import Generator
from Model.VAE import SpeakerIdentity
from Kode.dataload import DataLoad
from Kode.Preprocessing_WAV import Preproccesing

data = DataLoad("../Kode/Data")
Prep = Preproccesing(n_mels = 80)
embeddings, labels = SpeakerIdentity(data.head(n = 1))
Specs = Prep.spec_Mel('../Kode/Data/p225/p225_001.wav', 'p225')

path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

G = Generator(32,256,512,32).eval().to(device)

M, x1 = torch.from_numpy(Prep.Mel_spectrogram.T).to(device).unsqueeze(0), torch.from_numpy(embeddings[0]).to(device).unsqueeze(0)

print(M.shape, x1.shape)

G(M, x1, x1)


