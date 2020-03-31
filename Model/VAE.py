import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
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
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Data = DataLoad("../Kode/Data")
encoder = load_encoder("../Real_Time_Voice_Cloning/encoder/pretrained.pt").float()

def SpeakerIdentity(Data):
    embedding = []
    labels = []
    for j in Data:
        for path in Data[j]:
            print(path)
            labels.append(j)
            y = librosa.load(path, sr = 16000)[0]
            y = preprocess_wav(y)
            embed = embed_utterance(y)
            embedding.append(embed)
    return np.array(embedding), labels

def EvalEmbedding(embedding, labels):
    X = TSNE(n_components=2 ).fit_transform(embedding)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels)
    plt.title("t-SNE")
    plt.show()


embedding, labels = SpeakerIdentity(Data.head(n = 9))
EvalEmbedding(embedding, labels)
