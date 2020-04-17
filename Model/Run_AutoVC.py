print("Hej flotte")
import os, sys
os.chdir(sys.path[0])

import torch
from Train_and_Loss import TrainLoader, loss, Train
from Kode.dataload import DataLoad2

print("Train Script started ...")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

data_path = "../data/VCTK-Data/VCTK-Corpus/wav48"
data, labels = DataLoad2(data_path)

print("Number of wav files: {:}".format(len(data)))
batch_size = 2
num_workers = 8
shuffle = True


n_steps = 100000
save_every = 10000
models_dir = "Training"
model_path_name = "trainedModel"
loss_path_name = "loss"

trainloader = TrainLoader(data, labels, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
Train(trainloader, n_steps, save_every, models_dir, model_path_name, loss_path_name)












