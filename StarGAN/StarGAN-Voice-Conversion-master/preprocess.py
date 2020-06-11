import librosa
import numpy as np
import os, sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess
import re

def resample(spk, origin_wavpath, target_wavpath):
	wavfiles = [i for i in os.listdir(join(origin_wavpath, spk)) if i.endswith(".wav")]
	for wav in wavfiles:
		folder_to = join(target_wavpath, spk)
		os.makedirs(folder_to, exist_ok=True)
		wav_to = join(folder_to, wav)
		wav_from = join(origin_wavpath, spk, wav)
		subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
	return 0

def resample_to_16k(origin_wavpath, target_wavpath, num_workers=1):
	os.makedirs(target_wavpath, exist_ok=True)
	spk_folders = os.listdir(origin_wavpath)
	print(f"> Using {num_workers} workers!")
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	for spk in spk_folders:
		futures.append(executor.submit(partial(resample, spk, origin_wavpath, target_wavpath)))
	result_list = [future.result() for future in tqdm(futures)]
	print(result_list)

def split_data(paths, test_size = 0.1):
    indices = np.arange(len(paths))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths

def get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test, sample_rate=16000, test_size = 0.1, prep_train = True, prep_test = True):
	paths = glob.glob(join(spk_fold_path, '*.wav'))
	spk_name = basename(spk_fold_path)
	train_paths, test_paths = split_data(paths, test_size)
	
           
	if prep_train:
		f0s = []
		coded_sps = []
		for wav_file in train_paths:
			f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
			f0s.append(f0)
			coded_sps.append(coded_sp)
		log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
		coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)
		np.savez(join(mc_dir_train, spk_name+'_stats.npz'), 
				log_f0s_mean=log_f0s_mean,
				log_f0s_std=log_f0s_std,
				coded_sps_mean=coded_sps_mean,
				coded_sps_std=coded_sps_std)
				
		for wav_file in tqdm(train_paths):
			wav_nam = basename(wav_file)
			f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
			normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
			np.save(join(mc_dir_train, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
			
	if prep_test:
		if prep_train is False:
			spk_stats = np.load(join(mc_dir_train, spk_name+'_stats.npz'))
			coded_sps_mean = spk_stats['coded_sps_mean']
			coded_sps_std = spk_stats['coded_sps_std']
		for wav_file in tqdm(test_paths):
			wav_nam = basename(wav_file)
			f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
			normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)
			np.save(join(mc_dir_test, wav_nam.replace('.wav', '.npy')), normed_coded_sp, allow_pickle=False)
	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()


	sample_rate_default = 16000
	origin_wavpath_default = "./data/VCTK-Corpus/wav48"
	target_wavpath_default = "./data/VCTK-Corpus/wav16"
	mc_dir_train_default = './data/mc/train'
	mc_dir_test_default = './data/mc/test'

	parser.add_argument("--sample_rate", type = int, default = 16000, help = "Sample rate.")
	parser.add_argument("--origin_wavpath", type = str, default = origin_wavpath_default, help = "The original wav path to resample.")
	parser.add_argument("--target_wavpath", type = str, default = target_wavpath_default, help = "The original wav path to resample.")
	parser.add_argument("--mc_dir_train", type = str, default = mc_dir_train_default, help = "The directory to store the training features.")
	parser.add_argument("--mc_dir_test", type = str, default = mc_dir_test_default, help = "The directory to store the testing features.")
	parser.add_argument("--num_workers", type = int, default = None, help = "The number of cpus to use.")
	parser.add_argument("--speakers", nargs="+", default=["all"], help = "The folder for speakers to preprocess, if 'all' is given, all speakers from origin_wavpath is used")
	parser.add_argument("--test_size", type=float, default=0.1, help="The percentage of data which should be but in the test folder")
	parser.add_argument("--resample", type=int, default=1, help="Whether to resample or not, as integer")
	parser.add_argument("--prep_train", type=int, default=1, help="Whether or not to preprocess the training data as integer")
	parser.add_argument("--prep_test", type=int, default=1, help="Whether or not to preprocess the test data as integer")
	parser.add_argument("--overwrite_old", type=int, default=1, help="Whether to overwrite existing speakers, as integer")

	argv = parser.parse_args()

	sample_rate = argv.sample_rate
	origin_wavpath = argv.origin_wavpath
	target_wavpath = argv.target_wavpath
	mc_dir_train = argv.mc_dir_train
	mc_dir_test = argv.mc_dir_test
	num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()
	speakers = argv.speakers
	test_size = argv.test_size if argv.test_size <= 1 else int(argv.test_size)
	resampling = bool(argv.resample)
	prep_train = bool(argv.prep_train)
	prep_test = bool(argv.prep_test)
	overwrite_old = bool(argv.overwrite_old)

	# The original wav in VCTK is 48K, first we want to resample to 16K
	if resampling:
		print("Resampling to 16000 Hz...")
		resample_to_16k(origin_wavpath, target_wavpath, num_workers=num_workers)

    # WE only use 10 speakers listed below for this experiment.
    #speaker_used = ['262', '272', '229', '232', '292', '293', '360', '361', '248', '251']
    #speaker_used = ['p'+i for i in speaker_used]
    
	if not overwrite_old:
		speak = []
		for f in os.listdir(mc_dir_test):
			s = re.search("(.*)_.*", f).group(1)
			if s not in speak:
				speak.append(s)
		speaker_used = [name for name in os.listdir(origin_wavpath) if (os.path.isdir(origin_wavpath+"/"+name)) and (name not in speak)] if speakers == ["all"] else speakers
	else:
		speaker_used = [name for name in os.listdir(origin_wavpath) if (os.path.isdir(origin_wavpath+"/"+name))] if speakers == ["all"] else speakers
	print(f"Speaker used: {speaker_used}")
	## Next we are to extract the acoustic features (MCEPs, lf0) and compute the corresponding stats (means, stds). 
	# Make dirs to contain the MCEPs
	os.makedirs(mc_dir_train, exist_ok=True)
	os.makedirs(mc_dir_test, exist_ok=True)

	num_workers = len(speaker_used) #cpu_count()
	print("number of workers: ", num_workers)
	executor = ProcessPoolExecutor(max_workers=num_workers)

	work_dir = target_wavpath
	# spk_folders = os.listdir(work_dir)
	# print("processing {} speaker folders".format(len(spk_folders)))
	# print(spk_folders)
	print("Preprocessing the speakers...")
	futures = []
	for spk in speaker_used:
		print(f"Preprocessing {spk}")
		spk_path = os.path.join(work_dir, spk)
		futures.append(executor.submit(partial(get_spk_world_feats, spk_path, mc_dir_train, mc_dir_test, sample_rate, test_size, prep_train, prep_test)))
	result_list = [future.result() for future in tqdm(futures)]
	print(result_list)
	sys.exit(0)

