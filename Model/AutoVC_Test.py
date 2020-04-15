import os, sys
os.chdir(".")
sys.path.append(os.path.abspath(os.curdir))
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Kode.dataload import DataLoad2
import seaborn as sns
from sklearn.manifold import TSNE
from Model.Speaker_encoder.inference import load_model as load_encoder
from Model.Speaker_encoder.audio import preprocess_wav
import librosa
from Model.Speaker_encoder.inference import embed_utterance
from Model.AutoVC.model_vc import Generator
from Kode.Preprocessing_WAV import Preproccesing
from sklearn.utils import resample
from sklearn.model_selection import ShuffleSplit
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(torch.cuda.is_available())
print(device)
#Data = DataLoad("../Kode/Data")
encoder = load_encoder("../Model/Speaker_encoder/pretrained_encoder.pt").float()
#Data, labels = DataLoad2("../Kode/Data")

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

#G = Generator(32,256,512,32).eval().to(device)
#g_checkpoint = torch.load('AutoVC/autovc.ckpt', map_location=torch.device(device))
#G.load_state_dict(g_checkpoint['model'])



def TrainLoader(Data,labels, batch_size = 2, shuffle = True, num_workers = 1):
    Data, labels = np.array(Data)[np.argsort(labels)], np.array(labels)[np.argsort(labels)]
    Prep = Preproccesing()
    embeddings = SpeakerIdentity(Data)
    emb = []
    for person in sorted(set(labels)):
        index = np.where(labels == person)
        X = embeddings[index]
        X = X.mean(0).unsqueeze(0).expand(len(index[0]), -1)
        emb.append(X)
    emb = torch.cat(emb, dim = 0)
    Mels = Prep.Mel_Batch(list(Data))

    C = torch.utils.data.DataLoader(ConcatDataset(Mels, emb), shuffle = shuffle,
                                    batch_size = batch_size, collate_fn = collate,
                                    num_workers = num_workers)

    return C

def collate(batch):
    batch = list(zip(*batch))
    lengths = torch.tensor([t.shape[1] for t in batch[0]])
    m = lengths.max()
    Mels = []
    for t in batch[0]:
        pad = torch.nn.ConstantPad2d((0, 0, 0, m - t.size(1)), 0)
        t = pad(t)
        Mels.append(t)
    Mels = torch.cat(Mels, dim = 0)
    embeddings = torch.cat([t.unsqueeze(0) for t in batch[1]], dim = 0)

    return [Mels, embeddings]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



