import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
import numpy as np
from tqdm import tqdm
from AutoVC.model_vc import Generator
from Speaker_encoder.inference import load_model as load_encoder
from Speaker_encoder.audio import preprocess_wav
import librosa
from Speaker_encoder.inference import embed_utterance
from Kode.dataload import DataLoad2
from Kode.Preprocessing_WAV import Preproccesing
import matplotlib.pyplot as plt
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

encoder = load_encoder("../Model/Speaker_encoder/pretrained_encoder.pt").float()


def SpeakerIdentity(Data):
    if type(Data) is str:
        Data = [[Data]]
    embedding = []
    print("Creating Speaker Embeddings...")
    for path in tqdm(Data):
        y = librosa.load(path, sr = 16000)[0]
        y = preprocess_wav(y)
        embed = embed_utterance(y)
        embedding.append(embed)

    return torch.from_numpy(np.array(embedding)).to(device)

def EvalEmbedding(embedding, labels):
    np.random.seed(2020)
    X = TSNE(n_components=2 ).fit_transform(embedding)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels)
    plt.title("t-SNE")
    plt.show()



#embedding = SpeakerIdentity(Data)
#EvalEmbedding(embedding, labels)
#P = Preproccesing()

G = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('Models/trainedModel.pt', map_location=torch.device(device))
G.load_state_dict(g_checkpoint["model_state"])

Data, labels = DataLoad2("Kode/Data")

Data, labels = Data[:20], labels[:20]
P = Preproccesing()
X = P.Mel_Batch(Data[0])[0]
Y = P.Mel_Batch(Data[10])[0]
c_org = SpeakerIdentity(Data[:10])
c_org = c_org.mean(dim = 0).unsqueeze(0)

c_trg = SpeakerIdentity(Data[10:])
c_trg = c_trg.mean(dim = 0).unsqueeze(0)


mel, XY, codes = G(X, c_org, c_trg)
mel, XX, codes = G(X, c_org, c_org)

X = X.squeeze(0).detach().numpy()
XX = XX.squeeze(0).squeeze(0).detach().numpy()
XY = XY.squeeze(0).squeeze(0).detach().numpy()


plt.matshow(X)
plt.show()

plt.matshow(XX)
plt.show()

plt.matshow(XY)
plt.show()

plt.matshow(Y.squeeze(0).detach().numpy())
plt.show()





