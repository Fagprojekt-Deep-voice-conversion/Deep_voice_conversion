print("Hej flotte")
import os, sys
os.chdir(sys.path[0])

import torch
import torch.multiprocessing as mp
from AutoVC.model_vc import Generator
from Train_and_Loss import TrainLoader, loss, Train
from Kode.dataload import DataLoad2

print("Train Script started ...")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

data_path = "../data/VCTK-Data/VCTK-Corpus/wav48"
data, labels = DataLoad2(data_path)
data, labels = data[:7000], labels[:7000]
print("Number of wav files: {:}".format(len(data)))
batch_size = 2
num_workers = 0

shuffle = True
pin_memory = False 
seed = 20
n_steps = 100000
save_every = 10000
models_dir = "Models"
model_path_name = "trainedModel30k"
loss_path_name = "loss30k"





trainloader = TrainLoader(data, labels, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)




model = Generator(32, 256, 512, 32).eval().to(device)
g_checkpoint = torch.load('AutoVC/autovc.ckpt', map_location=torch.device(device))
model.load_state_dict(g_checkpoint['model'])
model.share_memory()

Train(model, trainloader, n_steps, save_every, models_dir, model_path_name, loss_path_name)













