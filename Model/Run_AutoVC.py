print("Hej flotte")
print("""
             .-^^^-.
            /       \
            \       /
     .-^^^-.-`.-.-.<  _
    /      _,-\ O_O-_/ )  ~champagne, monsieur?
    \     / ,  `   . `|
     '-..-| \-.,__~ ~ /          .,
           \ `-.__/  /          /"/
          / `-.__.-\`-._    ,",' ;
         / /|    ___\-._`-.; /  ./-.  
        ( ( |.-"`   `'\ '-( /  //.-' 
         \ \/    {}{}  |   /-. /.-'
          \|           /   '..' 
           \        , /
           ( __`;-;'__`)
           `//'`   `||`
          _//       ||
  .-"-._,(__)     .(__).-""-.
 /          \    /           \
 \          /    \           /
  `'-------`      `--------'`
                    """)
# Imports
import os, sys
os.chdir(sys.path[0])
import torch
import torch.multiprocessing as mp
from AutoVC.model_vc import Generator
from Train_and_Loss import TrainLoader, loss, Train
from Kode.dataload import DataLoad2

import argparse

# Run

if __name__ == "__main__":
	#parser = argparse.ArgumentParser()
	
    # Set device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(f"Train Script started on {device} ...")
	
	#Arguments
	
	### Data
	data_path = "Kode/Data"
	data, labels = DataLoad2(data_path)
	data, labels = data[:10], labels[:10]
	
	batch_size = 2
	num_workers = 0

	shuffle = True
	pin_memory = False 
	seed = 20
	n_steps = 100#000
	save_every = 10#000
	models_dir = "Models"
	model_path_name = "trainedModeltest"
	loss_path_name = "loss"
	
	print("Number of wav files: {:}".format(len(data)))
	
	
	# execute
	trainloader, corrupted = TrainLoader(data, labels, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)
	print("Number of corrupted files: ", len(corrupted))
	model = Generator(32, 256, 512, 32).eval().to(device)
	g_checkpoint = torch.load('AutoVC/autovc.ckpt', map_location=torch.device(device))
	model.load_state_dict(g_checkpoint['model'])
	model.share_memory()

	Train(model, trainloader, n_steps, save_every, models_dir, model_path_name, loss_path_name)
