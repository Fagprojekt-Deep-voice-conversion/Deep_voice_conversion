import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from Kode.dataload import DataLoad
import seaborn as sns
from sklearn.manifold import TSNE
from Model.encoder.inference import load_model as load_encoder
from Model.encoder.audio import preprocess_wav
import librosa
from Model.encoder.inference import embed_utterance
from Kode.Preprocessing_WAV import Preproccesing
from Model.VAE import SpeakerIdentity, EvalEmbedding
from Model.Encoder import Encoder as contEnc

import os, sys

os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))

path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Data = DataLoad("../Kode/Data")
encoder = load_encoder("../Model/encoder/pretrained.pt").float()

embedding, labels, paths = SpeakerIdentity(Data.head(n = 9))
#EvalEmbedding(embedding, labels)

CE = contEnc(32,256,32)
Prepros = Preproccesing()
#spectro = Prepros.spec_Mel(paths,labels)
spectro = torch.tensor(Prepros.spec_Mel(paths[0],"p225"))
embed = torch.tensor(embedding[0])
test = CE.forward(spectro,embed)