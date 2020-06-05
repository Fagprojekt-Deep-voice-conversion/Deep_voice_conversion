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
from Generator_autoVC.model_vc import Generator
from Train_and_Loss import TrainLoader, loss, Train
from dataload import DataLoad2
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
	model_path_name = "trainedModeltest"
	loss_path_name = "loss"
	
	print("Number of wav files: {:}".format(len(data)))
	"""

	### Init
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=20, help='Seed to use')
	
	### Trainloader
	parser.add_argument('--data_path', type=str, default='../data/VCTK-Data/VCTK-Corpus/wav48/', help='Path to the training data.')
	parser.add_argument('--num_train_data', type=int, default=None, help='Number of training samples to use. Will not be taken random, but from the beginning.')
	parser.add_argument('--batch_size', type=int, default=2, help='The batch size used for training')
	parser.add_argument('--num_workers', type=int, default=0, help='The number of workers used when loading data for the trainloader')
	parser.add_argument('--shuffle', action='store_true', default=True, help='Whether to shuffle the data or not when using it for training')
	parser.add_argument('--pin_memory', action='store_false', default=False, help='Whether to pin the memory or not')

	### Model
	parser.add_argument('--pretrained_model_path', type=str, default='Models/AutoVC/autovc_origin.ckpt', help='Path to pretrained model')
	
	### Train
	parser.add_argument('--init_lr', type=float, default=1e-3, help='Initial Learning Rate')
	parser.add_argument('--n_steps', type=int, default=100000, help='Number of training steps')
	parser.add_argument('--save_every', type=int, default=10000, help='Number of steps between each save of the model')
	parser.add_argument('--models_dir', type=str, default='Models/AutoVC', help="Directory to save the training results in")
	parser.add_argument('--vocoder', type=str, default='wavernn', help="Sets vocoder training parameters")
	parser.add_argument('--loss_dir', type=str, default='Models', help="Directory to save loss curve in")
	parser.add_argument('--model_path_name', type=str, default='trained_model', help='Name of the trained model')
	parser.add_argument('--loss_path_name', type=str, default='loss', help='Name of file containing loss values')
	
	# execute
	"""
	trainloader, corrupted = TrainLoader(data, labels, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)
	print("Number of corrupted files: ", len(corrupted))
	model = Generator(32, 256, 512, 32).eval().to(device)
	g_checkpoint = torch.load('AutoVC/autovc.ckpt', map_location=torch.device(device))
	model.load_state_dict(g_checkpoint['model'])
	model.share_memory()

	Train(model, trainloader, n_steps, save_every, models_dir, model_path_name, loss_path_name)
	"""
	config = parser.parse_args()
	
	### Initialise
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	
	### Run trainloader
	
	data, labels = DataLoad2(config.data_path)

	if config.num_train_data is not None:
		data, labels = data[:config.num_train_data ], labels[:config.num_train_data ]
	print("Number of wav files: {:}".format(len(data)))
	trainloader, uncorrupted = TrainLoader(data, labels, batch_size = config.batch_size, shuffle = config.shuffle, 
								num_workers = config.num_workers, pin_memory = config.pin_memory, vocoder = config.vocoder)
	
	### Make model
	model = Generator(32, 256, 512, 32).eval().to(device)
	#g_checkpoint = torch.load(config.pretrained_model_path, map_location=torch.device(device))
	#model.load_state_dict(g_checkpoint['model'])
	#model.share_memory()
	
	### Train model
	np.random.seed(config.seed)
	Train(model, trainloader, config.init_lr, config.n_steps, config.save_every, config.models_dir, config.loss_dir, config.model_path_name, config.loss_path_name)

