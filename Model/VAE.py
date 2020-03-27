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
from scipy.linalg import svd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as  KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as cos
import umap
from sklearn.manifold import TSNE
#import umap.plot
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
"""
try:
    checkpoint = torch.load("WaveNetVC_pretrained.pth", map_location = torch.device(device))
except FileNotFoundError:
    checkpoint = torch.load("Y:/Desktop/fagprojekt/WaveNetVC_pretrained.pth", map_location = torch.device(device))
"""
#WaveNet = build_model().to(device)
#WaveNet.load_state_dict(checkpoint["state_dict"])

Data = DataLoad("../Kode/Data")

from Real_Time_Voice_Cloning.encoder.inference import load_model

model = load_model("../Real_Time_Voice_Cloning/encoder/pretrained.pt").float()



def speaker_identity(Data, model, window_widt, step, sample_rate):
    n_fft = int(sample_rate * window_widt / 1000)
    hop_size = int(sample_rate * step / 1000)
    Prep = Preproccesing(sr = sample_rate, n_fft = n_fft, hop = hop_size)
    Prep.spec_Mel(Data.head(n=9).to_numpy().T.tolist(), Data.columns.tolist())

    Slices, b = Prep.slicer(160)
    Slices = torch.from_numpy(Slices).float()
    embedding = model.forward(torch.clone(Slices).detach())
    embedding = embedding.detach().numpy()
    speaker_id = np.array([np.mean(embedding[B], axis = 0) for B in b])
    return speaker_id, Prep.labels
embedding, labels = speaker_identity(Data, model, 25, 10, 16000)


print(np.shape(embedding))
U, s, vh = svd(embedding)
print(np.shape(vh), np.shape(s))
PC = vh
print(np.shape(PC), np.shape(embedding))
Z = embedding @ PC.T

print(np.shape(Z))


label = labels
cols = ["green", "red", "orange","blue", "magenta", "skyblue", "navy", "darkgreen", "black", "orangered"]
di = {p: cols[i] for i,p in enumerate(label)}

plt.plot(np.cumsum(s**2)/np.sum(s**2))
plt.show()


mu = np.array([np.mean(embedding[slice(i*9, i*9 + 9)], axis = 0) for i in range(10)])
KNN = KNN(n_neighbors=1)
KNN = KNN.fit(mu, np.arange(10))
ye = KNN.predict(embedding)
print("KNN accuracy:" ,accuracy_score(np.sort([0,1,2,3,4,5,6,7,8,10]*9), ye))


sns.scatterplot(Z[:,0], Z[:,1], hue = sorted(label*9), palette=di)
plt.title("PCA")
plt.show()


m = umap.UMAP().fit(embedding)
sns.scatterplot(m.embedding_[:,0], m.embedding_[:,1], hue = sorted(label*9), palette=di)

plt.title("UMAP")
plt.show()


X = TSNE(n_components=2).fit_transform(embedding)
sns.scatterplot(X[:,0], X[:,1], hue = sorted(label*9), palette=di)
plt.title("t-SNE")
plt.show()





