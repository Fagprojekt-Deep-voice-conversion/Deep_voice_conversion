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

data_path = "../Kode/Data"
data, labels = DataLoad2(data_path)

print("Number of wav files: {:}".format(len(data)))
batch_size = 2
num_workers = 0
shuffle = True


n_steps = 2
save_every = 10000
models_dir = "Models"
model_path_name = "trainedModel"
loss_path_name = "loss"

trainloader = TrainLoader(data[:5], labels[:5], batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
Train(trainloader, n_steps, save_every, models_dir, model_path_name, loss_path_name)












