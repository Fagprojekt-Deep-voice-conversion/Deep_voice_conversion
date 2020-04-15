"""
The loss and training script for AutoVC (home made but inspired by AutoVC paper)

"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from Model.AutoVC.model_vc import Generator
from Model.AutoVC_Test import SpeakerIdentity
from Kode.dataload import DataLoad2
from Model.AutoVC_Test import TrainLoader
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
trainloader = TrainLoader(data, labels)

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
    out_decoder, out_post, codes = output[0].squeeze(1), output[1].squeeze(1), output[2]
    X, c_org = target[0], target[1]
    ReconCodes = G(out_post, c_org, None)
    """ 
    Reconstruction error: 
        The mean of the squared p2 norm of (Postnet outputs - Original Mel Spectrograms)
    """
    err_reconstruct  = torch.dist(X, out_post, 2)
    """
    Prenet Reconstruction error
        The mean of the squared p2 norm of (Decoder outputs - Original Mel Spectrograms)
    """
    err_reconstruct0 = torch.dist(X, out_decoder, 2)

    """
    Content reconstruction Error
        The mean of the p1 norm of (Content codes of postnet output - Content codes)
    """
    err_content      = torch.dist(ReconCodes, codes, 1)

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
K = 2
for j in range(K):
    print("epoch {:} out of {:}".format(j+1, K))
    for batch in tqdm(trainloader):

        X, c_org = batch[0], batch[1]

        """ Outputs and loss"""
        mel, post, codes = G(X, c_org, c_org)
        error = loss([mel, post, codes], [X, c_org])

        """ Zeros the gradient for every step """
        """ Computes gradient and do optimiser step"""
        G.zero_grad()
        error.backward()
        optimiser.step()

        """ Append current error to L for plotting """
        r = error.detach().numpy()
        L.append(r)
plt.plot(L)
plt.show()

post = post.squeeze(1).detach().numpy()
plt.matshow(post)
plt.show()


