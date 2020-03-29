import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
from Model.AutoVC.autovc_master.synthesis import build_model
from Model.AutoVC.autovc_master.synthesis import wavegen
from Kode.Preprocessing_WAV import Preproccesing
import Kode.Data as Data
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
from Model.SpeakerEncoder import SpeakerEncoder
from ge2e import GE2ELoss
from Kode.dataload import DataLoad
path = sys.path[0]
os.chdir(path)



# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

try:
    checkpoint = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))
except FileNotFoundError:
    checkpoint = torch.load("Y:/Desktop/fagprojekt/WaveNetVC_pretrained.pth", map_location = torch.device(device))

#WaveNet = build_model().to(device)
#WaveNet.load_state_dict(checkpoint["state_dict"])

Data = DataLoad("../Kode/Data")
# Load Spectrogram examples of speech
#spect_vc2 = pickle.load(open('AutoVC/autovc_master/metadata.pkl', 'rb'))

Prep = Preproccesing()
#Prep.spec_Mel(Data['p227'].tolist())
#Prep.ShowSpec()


model = SpeakerEncoder()
criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#model.PartialSlicer(Data[["p225","p225"]].iloc[:10,:].to_numpy().T.tolist())

Prep.spec_Mel("p227_001.wav", "Hej")
Prep.ShowSpec()

from resemblyzer.voice_endcoder import VoiceEncoder

m = VoiceEncoder()
a, b = m.compute_partial_slices(2020202, 0.625, 0.5)

print(b)