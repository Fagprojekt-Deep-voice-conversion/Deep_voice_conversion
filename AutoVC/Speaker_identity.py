import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
import numpy as np
from tqdm import tqdm
from Speaker_encoder.inference import load_model as load_encoder
from Speaker_encoder.audio import preprocess_wav
import librosa
from Speaker_encoder.inference import embed_utterance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
encoder = load_encoder("Models/SpeakerEncoder/SpeakerEncoder.pt").float()

def SpeakerIdentity(Data):
    if type(Data) is str:
        Data = [Data]
    embedding = []
    uncorrupted = np.ones(len(Data), dtype = np.bool)
    print("Creating Speaker Embeddings...")
    for i,path in tqdm(enumerate(Data)):
        try:
            y = librosa.load(path, sr = 16000)[0]
            y = preprocess_wav(y)
            embed = embed_utterance(y)
            embedding.append(embed)
        except:
            uncorrupted[i] = False


    return torch.from_numpy(np.array(embedding)).to(device), uncorrupted

def EvalEmbedding(embedding, labels):
    np.random.seed(2020)
    X = TSNE(n_components=2 ).fit_transform(embedding)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels)
    plt.title("t-SNE")
    plt.show()
