import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import torch
print(sys.path[0])
from vocoder.WaveRNN_model import WaveRNN

from hparams import hparams_waveRNN as hp
import librosa
import numpy as np


def Generate(m, fpath, model):
    m = torch.tensor(m).unsqueeze(0)
    waveform = model.generate(m, batched = True, target = 11_000, overlap = 550, mu_law= False)
    librosa.output.write_wav(fpath + '.wav', np.asarray(waveform), sr = hp.sample_rate)

