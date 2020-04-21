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
import numpy as np

import argparse

# Run

if __name__ == "__main__":	
    # Set device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(f"Train Script started on {device} ...")
	
	#Arguments
	
	### Data
	"""
	data_path = "../data/VCTK-Data/VCTK-Corpus/wav48"
	data, labels = DataLoad2(data_path)
	data, labels = data[:20], labels[:20]
	
	batch_size = 2
	num_workers = 0

	shuffle = True
	pin_memory = False 
	seed = 20
	n_steps = 100#000
	save_every = 10#000
	models_dir = "Models"
	model_path_name = "trainedModel"
	loss_path_name = "loss"
	"""
	### Init
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=20, help='Seed to use')
	#init_config = init_parser.parse_args()
	
	### Trainloader
	#tl_parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default="../data/VCTK-Data/VCTK-Corpus/wav48", help='Path to the training data.')
	parser.add_argument('--num_train_data', type=int, default=None, help='Number of training samples to use. Will not be taken random, but from the beginning.')
	parser.add_argument('--batch_size', type=int, default=2, help='The batch size used for training')
	parser.add_argument('--num_workers', type=int, default=0, help='The number of workers used when loading data for the trainloader')
	parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the data or not when using it for training')
	parser.add_argument('--pin_memory', type=bool, default=False, help='Whether to pin the memory or not')
	#tl_config = tl_parser.parse_args()

	### Model
	#model_parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained_model_path', type=str, default='AutoVC/autovc.ckpt', help='Path to pretrained model')
	#model_config = model_parser.parse_args()	
	
	### Train
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_steps', type=int, default=100000, help='Number of training steps')
	parser.add_argument('--save_every', type=int, default=10000, help='Number of steps between each save of the model')
	parser.add_argument('--models_dir', type=str, default="Models", help="Directory to save the training results in")
	parser.add_argument('--model_path_name', type=str, default="trained_model", help='Name of the trained model')
	parser.add_argument('--loss_path_name', type=str, default="loss", help='Name of file containing loss values')
	#train_config = train_parser.parse_args()
	
	# execute
	config = parser.parse_args()
	init_config = config
	tl_config = config
	model_config = config
	train_config = config
	
	### Initialise
	np.random.seed(init_config.seed)
	torch.manual_seed(init_config.seed)
	
	### Run trainloader
	data, labels = DataLoad2(tl_config.data_path)
	if tl_config.num_train_data is not None:
		data, labels = data[:tl_config.num_train_data ], labels[:tl_config.num_train_data ]
	print("Number of wav files: {:}".format(len(data)))
	trainloader = TrainLoader(data, labels, batch_size = tl_config.batch_size, shuffle = tl_config.shuffle, 
								num_workers = tl_config.num_workers, pin_memory = tl_config.pin_memory)
	
	### Make model
	model = Generator(32, 256, 512, 32).eval().to(device)
	g_checkpoint = torch.load(model_config.pretrained_model_path, map_location=torch.device(device))
	model.load_state_dict(g_checkpoint['model'])
	model.share_memory()
	
	### Train model
	Train(model, trainloader, train_config.n_steps, train_config.save_every, train_config.models_dir, train_config.model_path_name, train_config.loss_path_name)














