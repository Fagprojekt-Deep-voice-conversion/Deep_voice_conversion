
import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
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


# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_, spect_vc = sm("p227_001.wav")

try:
    checkpoint = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))
except FileNotFoundError:
    checkpoint = torch.load("Y:/Desktop/fagprojekt/WaveNetVC_pretrained.pth", map_location = torch.device(device))

model = build_model().to(device)
model.load_state_dict(checkpoint["state_dict"])

# Load Spectrogram examples of speech
spect_vc2 = pickle.load(open('AutoVC/autovc_master/metadata.pkl', 'rb'))
_, spect_vc = sm("p225_001.wav")
#name = "Reconstruction_Test"

# Manually Crop of silence
c = spect_vc2[0][2]

plt.matshow(spect_vc.T[20:140,:])
plt.title("Vores forsøg")
plt.show()


plt.matshow(c)
plt.title("Deres")
plt.show()
# Convert to waveform using WaveNet Vocoder (pretrained)
waveform = wavegen(model, c=spect_vc.T[20:140,:])
#Sound = librosa.feature.inverse.mel_to_stft(spect_vc)

# Short Time Fourier Transformation of signal

#plt.plot(waveform)
#plt.show()


librosa.output.write_wav(name+'.wav', b, sr=16000)
