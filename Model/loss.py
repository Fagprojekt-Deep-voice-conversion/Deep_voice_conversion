import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from Model.AutoVC.autovc_master.model_vc import Generator
from Model.VAE import SpeakerIdentity
from Kode.dataload import DataLoad2
from Kode.Preprocessing_WAV import Preproccesing
path = sys.path[0]
os.chdir(path)

# Check for GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Prep = Preproccesing(n_mels = 80)

data, labels = DataLoad2("../Kode/Data")



trainloader = torch.utils.data.DataLoader(data, batch_size = 2, shuffle = True)

G = Generator(32,256,512,32).eval().to(device)


def loss(output, target, mu = 1, lambd = 1):
    output = [*zip(*output)]
    err_reconstruct  = [torch.dist(output_post, target[0][i], 2) ** 2 for i, output_post in enumerate(output[1])]
    err_reconstruct0 = [torch.dist(output_mel, target[0][i], 2)**2 for i, output_mel in enumerate(output[0])]
    err_content      = [torch.dist(G(output[1][i].squeeze(0), target[1][i].unsqueeze(0), None), content_codes, 1)
                        for i, content_codes in enumerate(output[2])]

    err_reconstruct  = torch.tensor(err_reconstruct, requires_grad = True).mean()
    err_reconstruct0 = torch.tensor(err_reconstruct0, requires_grad = True).mean()
    err_content      = torch.tensor(err_content, requires_grad = True).mean()

    return err_reconstruct + mu * err_reconstruct0 + lambd * err_content



optimiser = torch.optim.Adam(G.parameters(), lr = 1e-3)

L = []
K = 1
for j in range(K):
    for i, batch in enumerate(trainloader, 0 ):
        print((i + (j * len(trainloader))) / (len(trainloader)*K) * 100, ' %')

        optimiser.zero_grad()
        Mel = Prep.Mel_Batch(batch)
        embedding, label = SpeakerIdentity([batch])

        outputs = [G(X, embedding[i].unsqueeze(0), embedding[i].unsqueeze(0)) for i, X in enumerate(Mel)]
        error = loss(outputs, (Mel, embedding))
        error.backward()

        optimiser.step()

        r = error.detach().numpy()
        L.append(r)


plt.plot(L)
plt.show()





embeddings, labels = SpeakerIdentity(data)
Specs = Prep.spec_Mel('../Kode/Data/p225/p225_009.wav', 'p225')


G = Generator(32,256,512,32).eval().to(device)

M, x1 = torch.from_numpy(Prep.Mel_spectrogram.T).to(device).unsqueeze(0), torch.from_numpy(embeddings[0]).to(device).unsqueeze(0)

print(M.shape, x1.shape)
out, post, content = G(M, x1, x1)
print(post.squeeze(0).shape)
print(torch.dist(M, post.squeeze(0), 2)**2)
print(G.Content(post))


def loss(output, target, mu = 1, lambd = 1):
    output_post = output[1]
    output = output[0]
    content_codes = output[2]
    encoder = G.Content()
    err_reconstruct  = torch.mean(torch.dist(output_post, target, 2)**2, 0)
    err_reconstruct0 = torch.mean(torch.dist(output_mel, target, 2)**2, 0)
    err_content      = torch.mean(torch.dist(encoder(output_post), content_codes, 1), 0)

    return err_reconstruct + mu * err_reconstruct0 + lambd * err_content

