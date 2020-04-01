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
from Kode.Preprocessing_WAV import Preproccesing
#path = sys.path[0]
#os.chdir(path)

# Check for GPU
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")

#Data = DataLoad("../Kode/Data")
#encoder = load_encoder("../Model/encoder/pretrained.pt").float()

def SpeakerIdentity(Data):
    if type(Data) is str:
        Data = [[Data]]
    embedding = []
    labels = []
    paths = []
    for j in Data:
        for path in Data[j]:
            print(path)
            paths.append(path)
            labels.append(j)
            y = librosa.load(path, sr = 16000)[0]
            y = preprocess_wav(y)
            embed = embed_utterance(y)
            embedding.append(embed)
    return np.array(embedding), labels, paths

def EvalEmbedding(embedding, labels):
    np.random.seed(2020)
    X = TSNE(n_components=2 ).fit_transform(embedding)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels)
    plt.title("t-SNE")
    plt.show()


#embedding, labels = SpeakerIdentity(Data.head(n = 9))
#EvalEmbedding(embedding, labels)
"""
from Real_Time_Voice_Cloning.vocoder.inference import load_model, infer_waveform
from Real_Time_Voice_Cloning.synthesizer.inference import Synthesizer
load_model("../Real_Time_Voice_Cloning/vocoder/pretrained.pt")
path = Data.iloc[0,0]
y = librosa.load(path)[0]
y = Synthesizer.load_preprocess_wav(path)
y = Synthesizer.make_spectrogram(y)

plt.matshow(y)
plt.show()

x = infer_waveform(y)
plt.plot(x[:25000])
plt.show()

librosa.output.write_wav("test.wav", x)
"""
