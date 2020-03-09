import torch
import os
import sys
from Model.AutoVC.autovc_master.synthesis import build_model
from Model.AutoVC.autovc_master.synthesis import wavegen
from Model.AutoVC.autovc_master.WavtoSpec import spec_Mel as sm
from
path = sys.path[0]
os.chdir(path)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# spect_vc = pickle.load(open('results.pkl', 'rb'))
# checkpoint = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))
# model = build_model().to(device)
# model.load_state_dict(checkpoint["state_dict"])

# for spect in spect_vc:
#     name = spect[0]
#     c = spect[1]
#     print(name)
#     waveform = wavegen(model, c=c)   
#     librosa.output.write_wav(name+'.wav', waveform, sr=16000)
