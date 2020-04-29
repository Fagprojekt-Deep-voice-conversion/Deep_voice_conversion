import matplotlib.pyplot as plt
import pickle
import os, sys; os.chdir(sys.path[0])
from Generator_autoVC.model_vc import Generator
import torch
from dataload import DataLoad2
from Preprocessing_WAV import Preproccesing
from Speaker_identity import SpeakerIdentity
import seaborn as sns

data, labels = DataLoad2("Test_Data")
data = data[:20]
def Conversion(source, target, model):
    P = Preproccesing()
    S, _  = P.Mel_Batch(source)
    T, _  = P.Mel_Batch(target)

    s_emb, _ = SpeakerIdentity(source)
    t_emb, _ = SpeakerIdentity(target)

    S, T = S[0], T[0]
    mel, post, codes = model(S, s_emb, t_emb)
    S = S.squeeze(0).detach().numpy()

    post = post.squeeze(0).squeeze(0).detach().numpy()
    plt.matshow(S)
    plt.matshow(post)
    plt.show()



s = data[2]
t = data[10]


model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/trainedModel30k.pt", map_location=torch.device("cpu"))
model.load_state_dict(g_checkpoint['model_state'])
Conversion(s,t, model)
"""
data, labels = DataLoad2("Test_Data")
data = data[:20]
mels, _ = P.Mel_Batch(data)
emb, _ = SpeakerIdentity(data)

c_org = emb[:10].mean(0).unsqueeze(0)
c_trg = emb[10:].mean(0).unsqueeze(0)

# print(c_org.shape, c_trg.shape)

X = mels[1]
Y = mels[11]
print(X.shape, Y.shape, c_org.shape, c_trg.shape, type(X))
mel, XX, code = model(X, c_org, c_org)
mel, XY, code = model(X, c_org, c_trg)
mel, YY, code = model(Y, c_trg, c_trg)
mel, YX, code = model(Y, c_trg, c_org)

X = mels[1].squeeze(0).numpy()
Y = mels[11].squeeze(0).numpy()
XX = XX.squeeze(0).squeeze(0).detach().numpy()
XY = XY.squeeze(0).squeeze(0).detach().numpy()
YY = YY.squeeze(0).squeeze(0).detach().numpy()
YX = YX.squeeze(0).squeeze(0).detach().numpy()
fig = plt.figure(figsize=(20,20))
fig.add_subplot(2,3,1)
sns.heatmap(X)
plt.title("X")
fig.add_subplot(2,3,2)
sns.heatmap(XX)
plt.title("X to X")
fig.add_subplot(2,3,3)
sns.heatmap(XY)
plt.title("X to Y")
fig.add_subplot(2,3,4)
sns.heatmap(Y)
plt.title("Y")
fig.add_subplot(2,3,5)
sns.heatmap(YY)
plt.title("Y to Y")
fig.add_subplot(2,3,6)
sns.heatmap(YX)
plt.title("Y to X")


plt.show()





"""



