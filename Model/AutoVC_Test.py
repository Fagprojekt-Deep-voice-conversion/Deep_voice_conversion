import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
import numpy as np
import matplotlib.pyplot as plt
from Kode.dataload import DataLoad2
import seaborn as sns
from sklearn.manifold import TSNE
from Model.Speaker_encoder.inference import load_model as load_encoder
from Model.Speaker_encoder.audio import preprocess_wav
import librosa
from Model.Speaker_encoder.inference import embed_utterance
from Kode.Preprocessing_WAV import Preproccesing
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Data = DataLoad("../Kode/Data")
encoder = load_encoder("../Model/Speaker_encoder/pretrained_encoder.pt").float()
Data, labels = DataLoad2("../Kode/Data")

def SpeakerIdentity(Data):
    if type(Data) is str:
        Data = [[Data]]

    for path in Data:
        print(path)
        y = librosa.load(path, sr = 16000)[0]
        y = preprocess_wav(y)
        embed = embed_utterance(y)
        embedding.append(embed)
    return torch.from_numpy(np.array(embedding))

def EvalEmbedding(embedding, labels):
    np.random.seed(2020)
    X = TSNE(n_components=2 ).fit_transform(embedding)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels)
    plt.title("t-SNE")
    plt.show()

X = Data[0]

print(X)
#embedding = SpeakerIdentity(Data)
#EvalEmbedding(embedding, labels)

