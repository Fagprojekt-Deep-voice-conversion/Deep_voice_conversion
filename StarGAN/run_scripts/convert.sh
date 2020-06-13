#!/bin/bash
### Set directory

if [ "$1" = "setup" ]
then
	BASE_DIRECTORY=$(echo "$PWD" | cut -d "/" -f2)
	if [ $BASE_DIRECTORY = "work1" ]
	then 
		run_dir=/work1/s183920/Deep_voice_conversion/StarGAN/run_scripts
	else
		run_dir=~/Desktop/Deep_voice_conversion/StarGAN/run_scripts/
	fi
	
	

	### Run setup
	sh setup.sh $run_dir || exit 1

fi

datadir=/work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/30min
moddir=/work1/s183920/Deep_voice_conversion/StarGAN
modname=30min_seed2000
conv_dir=/work1/s183920/Deep_voice_conversion/data/results/StarGAN/$modname/converted 
steps=190000


### Load modules
module load python3
module load cuda
source StarGAN-env/bin/activate

### Convert
### Must set the speakers that was used for preprocessing in script
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed 420 \
		--num_converted_wavs 1 \
		--resume_iters $steps \
		--src_spk trump \
		--trg_spk michelle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir \
		--files_to_convert random \
