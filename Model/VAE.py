import torch
import os
import sys
from Model.AutoVC.autovc_master.synthesis import build_model
from Model.AutoVC.autovc_master.synthesis import wavegen
from Kode.Preprocessing_WAV import spec_Mel as sm
from Kode.Preprocessing_WAV import play_Spec
import Kode.Data as Data
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
path = sys.path[0]
os.chdir(path)

spect_vc2 = pickle.load(open('AutoVC/autovc_master/metadata.pkl', 'rb'))


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_, spect_vc = sm("p227_001.wav")


checkpoint = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))
model = build_model().to(device)
model.load_state_dict(checkpoint["state_dict"])

name = "test"
c = spect_vc
print(np.shape(c), np.shape(spect_vc2[3][2]))
plt.matshow(spect_vc2[0][2])
plt.show()
plt.matshow(spect_vc.T)
plt.show()
#waveform = wavegen(model, c=c)
#Sound = librosa.feature.inverse.mel_to_stft(spect_vc)
#librosa.output.write_wav(name+'.wav', librosa.griffinlim(Sound), sr=16000)
#play_Spec(c.T)
#plt.show()
#print(np.shape(spect_vc2))