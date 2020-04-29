import matplotlib.pyplot as plt
import pickle
import os, sys; os.chdir(sys.path[0])
from Generator_autoVC.model_vc import Generator
import torch
from dataload import DataLoad2
from Preprocessing_WAV import Preproccesing
from Speaker_identity import SpeakerIdentity
import seaborn as sns
loss = pickle.load(open("Models/loss20k", "rb"))

plt.plot(loss)
plt.show()


model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/trained_model20k.pt", map_location=torch.device("cpu"))
model.load_state_dict(g_checkpoint['model_state'])

P = Preproccesing()
data, labels = DataLoad2("Test_Data")
data = data[:20]
mels, _ = P.Mel_Batch(data)
emb, _ = SpeakerIdentity(data)

c_org = emb[:10].mean(0).unsqueeze(0)
c_trg = emb[10:].mean(0).unsqueeze(0)

# print(c_org.shape, c_trg.shape)

X = mels[0]
Y = mels[10]
print(X.shape, Y.shape, c_org.shape, c_trg.shape, type(X))
mel, XX, code = model(X, c_org, c_org)
mel, XY, code = model(X, c_org, c_trg)
mel, YY, code = model(Y, c_trg, c_trg)
mel, YX, code = model(Y, c_trg, c_org)

X = mels[0].squeeze(0).numpy()
Y = mels[10].squeeze(0).numpy()
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









