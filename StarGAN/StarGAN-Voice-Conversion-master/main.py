import os
import argparse
from solver import Solver
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
from numpy.random import seed as seed_np
from torch import manual_seed as seed_t
import torch
import pickle


def str2bool(v):
    return v.lower() in ('true')

def main(config):
	# Set seed
	seed_np(config.seed)
	seed_t(config.seed)
	
    # For fast training.
	cudnn.benchmark = True

    # Create directories if not exist.
	if not os.path.exists(config.log_dir):
		os.makedirs(config.log_dir)
	if not os.path.exists(config.model_save_dir):
		os.makedirs(config.model_save_dir)
	if not os.path.exists(config.sample_dir):
		os.makedirs(config.sample_dir)
	if not os.path.exists(config.loader_dir):
		os.makedirs(config.loader_dir)
	
	# Data loader.
	# Train
	train_loader = get_loader(config.train_data_dir, config.batch_size, 'train', num_workers=config.num_workers)
	
	"""
	if os.path.exists(config.loader_dir+"/trainloader.pkl"):
		with open(config.loader_dir+"/trainloader.pkl", "rb") as f:
			train_loader = pickle.load(f)
		train_loader = pickle.loads(f)
	else:
		train_loader = get_loader(config.train_data_dir, config.batch_size, 'train', num_workers=config.num_workers)
		t = pickle.dumps(train_loader)
		with open(config.loader_dir+"/trainloader.pkl", "wb") as f:
			pickle.dump(t, f)
	"""
		
	# Test
	# test_loader = TestDataset(config.test_data_dir, config.wav_dir, src_spk='p262', trg_spk='p272')
	test_loader = TestDataset(config.test_data_dir, config.wav_dir, src_spk='louise', trg_spk=config.test_target)
	"""
	if os.path.exists(config.loader_dir+"/testloader.pkl"):
		with open(config.loader_dir+"/testloader.pkl", "rb") as f:
			test_loader = pickle.load(f)
		test_loader = pickle.loads(test_loader)
	else:
		test_loader = TestDataset(config.test_data_dir, config.wav_dir, src_spk='p262', trg_spk='p272')
		t = pickle.dumps(test_loader)
		with open(config.loader_dir+"/testloader.pkl", "wb") as f:
			pickle.dump(t, f)
	"""
			
	

	# Solver for training and testing StarGAN.
	solver = Solver(train_loader, test_loader, config)

	if config.mode == 'train':    
		solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Model configuration.
	parser.add_argument('--num_speakers', type=int, default=None, help='dimension of speaker labels')
	parser.add_argument('--lambda_cls', type=float, default=10, help='weight for domain classification loss')
	parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
	parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
	parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

	parser.add_argument("--model_name", type = str, default = "StarGAN_model")
	parser.add_argument('--test_target', type = str, default = "yangSMK")
	# parser.add_argument("--log_freq", type = int, default = 8)

	# Training configuration.
	parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
	parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
	parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
	parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
	parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
	parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
	parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
	parser.add_argument('--resume_from_max', type=int, default=1, help='whether to resume the training from the latest model, as int')

	# Test configuration.
	parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

	# Miscellaneous.
	parser.add_argument('--seed', type=int, default=420, help='seed used for training')
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	parser.add_argument('--use_tensorboard', type=str2bool, default=True)

	# Directories.
	#parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
	#parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
	parser.add_argument('--train_data_dir', type=str, default='././data/VCTK-Data/StarGAN/mc/train')
	parser.add_argument('--test_data_dir', type=str, default='././data/VCTK-Data/StarGAN/mc/test')
	#parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")
	parser.add_argument('--wav_dir', type=str, default="././data/VCTK-Data/StarGAN/wav16")
	parser.add_argument('--log_dir', type=str, default='./logs')
	parser.add_argument('--model_save_dir', type=str, default='./models')
	parser.add_argument('--sample_dir', type=str, default='./samples')
	parser.add_argument('--loss_name', type=str, default='loss', help='name to give the pickle files containing the losses')
	parser.add_argument('--loader_dir', type=str, default='././data/VCTK-Data/StarGAN/mc/loaders', help='path to saved loaders')

	# Step size.
	parser.add_argument('--log_step', type=int, default=10)
	parser.add_argument('--sample_step', type=int, default=1000)
	parser.add_argument('--model_save_step', type=int, default=1000)
	parser.add_argument('--lr_update_step', type=int, default=1000)

	config = parser.parse_args()
	print(config)
	main(config)
