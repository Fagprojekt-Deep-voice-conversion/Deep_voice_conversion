"""
The loss and training script for AutoVC (home made but inspired by AutoVC paper)

"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from Model.AutoVC.model_vc import Generator
from Model.AutoVC_Test import SpeakerIdentity
from Kode.dataload import DataLoad2
from Kode.Preprocessing_WAV import Preproccesing
path = sys.path[0]
os.chdir(path)

# Check for GPU
""" A bit of init stuff ... Checking for GPU and loading data """
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
Prep = Preproccesing(n_mels = 80)
data, labels = DataLoad2("../Kode/Data")


""" Creates batches for training. Batch size = 2 as in the Paper"""
trainloader = torch.utils.data.DataLoader(data, batch_size = 2, shuffle = True)

"""
Loads model of Generator network - Pretrained from AutoVC, hence transfer learning.
Hopefully this will speed training up... The speaker identity encoder is not the same as in AutoVC, which means it has to be trained...
"""
G = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('AutoVC/autovc.ckpt', map_location=torch.device(device))
G.load_state_dict(g_checkpoint['model'])


def loss(output, target, mu = 1, lambd = 1):
    """
    Loss function as proposed in AutoVC
    L = Reconstruction Error + mu * Prenet reconstruction Error + lambda * content Reconstruction error
    mu and lambda are set to 1 in the paper.

    params:
    outputs: as for now a simple try... A list of outputs of the batch:
        - batch_size * decoder_outputs
        - bacth_size * postnet_outputs
        - batch_size * content codes
    target: A list of the targets
        - batch_size * orignal Mel spectrograms
        - batch_size * original Speaker identiy embedding
    mu, lambda: model parameters mu and lambda

    returns the loss function as proposed in AutoVC
    """

    """ Zips output ... """
    output = [*zip(*output)]

    """ 
    Reconstruction error: 
        The mean of the squared p2 norm of (Postnet outputs - Original Mel Spectrograms)
    """
    err_reconstruct  = [torch.dist(output_post, target[0][i], 2) ** 2 for i, output_post in enumerate(output[1])]
    err_reconstruct = torch.tensor(err_reconstruct, requires_grad=True).mean()

    """
    Prenet Reconstruction error
        The mean of the squared p2 norm of (Decoder outputs - Original Mel Spectrograms)
    """
    err_reconstruct0 = [torch.dist(output_mel, target[0][i], 2)**2 for i, output_mel in enumerate(output[0])]
    err_reconstruct0 = torch.tensor(err_reconstruct0, requires_grad=True).mean()

    """
    Content reconstruction Error
        The mean of the p1 norm of (Content codes of postnet output - Content codes)
    """
    err_content      = [torch.dist(G(output[1][i].squeeze(0), target[1][i].unsqueeze(0), None), content_codes, 1)
                        for i, content_codes in enumerate(output[2])]
    err_content      = torch.tensor(err_content, requires_grad = True).mean()

    return err_reconstruct + mu * err_reconstruct0 + lambd * err_content


""" We use the Adam optimiser with init learning rate 1e-3"""
optimiser = torch.optim.Adam(G.parameters(), lr = 1e-3)



"""
Training.
K epochs with batch size 2. 

It does not seem like it is working... Problems with either loss function of optimiser / training...
No process is happening... I do not know how to fix it.
A second problem is the batch. We need a smart way to implement training on batches with spectrograms of 
not equal length...
"""
L = []
K = 1
for j in range(K):
    for i, batch in enumerate(trainloader, 0 ):
        """ Prints how far in the process in percentage """
        print((i + (j * len(trainloader))) / (len(trainloader)*K) * 100, ' %')

        """ Zeros the gradient for every step """
        optimiser.zero_grad()

        """ Creates Mel Spectrograms and Speaker Identity embedding. TODO: The spectrograms still do not have the right specifications ... """
        Mel = Prep.Mel_Batch(batch)
        embedding = SpeakerIdentity(batch)

        """ Outputs and loss"""
        outputs = [G(X, embedding[i].unsqueeze(0), embedding[i].unsqueeze(0)) for i, X in enumerate(Mel)]
        error = loss(outputs, (Mel, embedding))

        """ Computes gradient and do optimiser step"""
        error.backward()
        optimiser.step()

        """ Append current error to L for plotting """
        r = error.detach().numpy()
        L.append(r)
plt.plot(L)
plt.show()


