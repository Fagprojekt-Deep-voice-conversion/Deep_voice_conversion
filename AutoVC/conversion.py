import matplotlib.pyplot as plt
import pickle
import os, sys; os.chdir(sys.path[0])
from Generator_autoVC.model_vc import Generator
import torch
from dataload import DataLoad2
from Preprocessing_WAV import Preproccesing
from Speaker_identity import SpeakerIdentity
import seaborn as sns
from Train_and_Loss import TrainLoader
import numpy as np
import librosa



loss = pickle.load(open("Models/lossfull_set", "rb"))

x = np.convolve(np.asarray(loss), np.ones((100,))/100, mode='valid')
# plt.plot(loss, color = "C1")
# plt.grid()
# plt.show()

# plt.plot(x, color = "C1")
# plt.grid()
# plt.show()

data, labels = DataLoad2("Test_Data")
data,labels  = data[:2], labels[:2]

def Conversion(source, target, model, embs = None):
    P = Preproccesing()
    S1, _  = P.Mel_Batch(source)
    T1, _  = P.Mel_Batch(target)

    #s_emb, _ = SpeakerIdentity(source)
    #t_emb, _ = SpeakerIdentity(target)
    s_emb, t_emb = embs[0], embs[1]


    S, T = S1[0], T1[0]
    mel, ST1, codes = model(S, s_emb, t_emb)
    mel, SS, codes = model(S, s_emb, s_emb)
    mel, TS, codes = model(T, t_emb, s_emb)
    mel, TT, codes = model(T, t_emb, t_emb)
    S = S.squeeze(0).detach().numpy()
    T = T.squeeze(0).detach().numpy()
    ST = ST1.squeeze(0).squeeze(0).detach().numpy()
    SS = SS.squeeze(0).squeeze(0).detach().numpy()
    TS = TS.squeeze(0).squeeze(0).detach().numpy()
    TT = TT.squeeze(0).squeeze(0).detach().numpy()

    fig = plt.figure(figsize = (10,10))
    fig.add_subplot(2,3, 1)
    plt.imshow(S)
    plt.title("X")
    fig.add_subplot(2,3, 2)
    plt.imshow(SS)
    plt.title("X to X")
    fig.add_subplot(2,3, 3)
    plt.imshow(ST)
    plt.title("X to Y")
    fig.add_subplot(2,3, 4)
    plt.imshow(T)
    plt.title("Y")
    fig.add_subplot(2,3, 5)
    plt.imshow(TT)
    plt.title("Y to Y")
    fig.add_subplot(2,3, 6)
    plt.imshow(TS)
    plt.title("Y to X")
    plt.show()
    return ST1, S1, T1


#model = Generator(32, 256, 512, 32).eval().to("cpu")
#g_checkpoint = torch.load("Models/trained_model_fullsetaverage__step100k.pt", map_location=torch.device("cpu"))
#model.load_state_dict(g_checkpoint['model_state'])

#emb, _ = SpeakerIdentity(data)
#print(emb.shape)
#embs = (s_emb, t_emb) = emb[:10].mean(0).unsqueeze(0), emb[10:].mean(0).unsqueeze(0)

#ST, S , T = Conversion(s, t, model, embs)

""" Auto VC vocoder """

from vocoder.synthesis import build_model
from vocoder.synthesis import wavegen
model = build_model().to("cpu")
checkpoint = torch.load("vocoder/WaveNetVC_pretrained.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["state_dict"])

"""
waveform = wavegen(model, c = S[0].squeeze(0))
librosa.output.write_wav("source"+'.wav', waveform, sr=16000)
waveform = wavegen(model, c = T[0].squeeze(0))
librosa.output.write_wav("target"+'.wav', waveform, sr=16000)

waveform = wavegen(model, c = ST.squeeze(0).squeeze(0))
librosa.output.write_wav("conversion "+'.wav', waveform, sr=16000)

"""




import pickle
from Generator_autoVC.AutoVC_preprocessing import logmelspectrogram, trim, load_wav, _normalize
X = pickle.load(open("Outdated/metadata.pkl", "rb"))
p225 = X[0][2]
y = load_wav(data[0])
y = y / np.abs(y).max() * 0.999
y = trim(y)

#M = librosa.feature.melspectrogram(y, sr = 16000, n_fft = 1024, hop_length= 256, n_mels = 80, fmin =90, fmax = 7600, power = 1)

#M = librosa.amplitude_to_db(M, ref = 16)
#M = np.clip((M - (-100)) / -(-100), 0, 1)

P = Preproccesing()
M, _ = P.Mel_Batch(data)
waveform = wavegen(model, c = M[0].squeeze(0))
librosa.output.write_wav("test"+'.wav', waveform, sr=16000)




