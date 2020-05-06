"""
HEJ!
Det her script er stadig in the making og lidt Peters rodekasse :-)

Du vil nok ikke få meget ud af det pt.


"""












import matplotlib.pyplot as plt

import os, sys; os.chdir(sys.path[0])
from Generator_autoVC.model_vc import Generator
import torch
from dataload import DataLoad2
from Preprocessing_WAV import Preproccesing
from Speaker_identity import SpeakerIdentity

import librosa
from vocoder.synthesis import build_model
from vocoder.synthesis import wavegen

data, labels = DataLoad2("Test_Data")
data,labels  = data[:20], labels[:20]



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

s = data[1]
t = data[11]

model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/trained_model_fullaverage_step200k.pt", map_location=torch.device("cpu"))
model.load_state_dict(g_checkpoint['model_state'])

emb, _ = SpeakerIdentity(data)
print(emb.shape)
embs = (s_emb, t_emb) = emb[:10].mean(0).unsqueeze(0), emb[10:].mean(0).unsqueeze(0)

ST, S , T = Conversion(s, t, model, embs)

# Auto VC vocoder 


model = build_model().to("cpu")
checkpoint = torch.load("vocoder/WaveNetVC_pretrained.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["state_dict"])


waveform = wavegen(model, c = S[0].squeeze(0))
librosa.output.write_wav("source"+'.wav', waveform, sr=16000)
waveform = wavegen(model, c = T[0].squeeze(0))
librosa.output.write_wav("target"+'.wav', waveform, sr=16000)

waveform = wavegen(model, c = ST.squeeze(0).squeeze(0))
librosa.output.write_wav("conversion "+'.wav', waveform, sr=16000)






