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

### Load modules
# module load python3
module load cuda

source StarGAN-env/bin/activate

### Train model
#python ../StarGAN-Voice-Conversion-master/main.py
# datadir=/work1/s183920/Deep_voice_conversion_old/data/VCTK-Data/StarGAN/mc
# moddir=/work1/s183920/Deep_voice_conversion_old/StarGAN
# modname=base_test_old
datadir=/work1/s183920/Deep_voice_conversion/data/SMK3/mc
moddir=/work1/s183920/Deep_voice_conversion/StarGAN
modname=SMK3
steps=500000


python ../StarGAN-Voice-Conversion-master/main.py \
		--lambda_cls 10 \
		--lambda_rec 10 \
		--lambda_gp 10 \
		--sampling_rate 16000 \
		--batch_size 2 \
		--num_iters $steps \
		--num_iters_decay 100000 \
		--g_lr 0.0001 \
		--d_lr 0.0001 \
		--n_critic 5 \
		--beta1 0.5 \
		--beta2 0.999 \
		--test_iters 100000 \
		--seed 42 \
		--num_workers 1 \
		--mode train \
		--log_step 100 \
		--sample_step 5000 \
		--model_save_step 5000 \
		--lr_update_step 1000 \
		--train_data_dir $datadir/train \
		--test_data_dir $datadir/test \
		--loader_dir $datadir/loader \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--sample_dir $moddir/samples/$modname \
		--model_save_dir $moddir/models/$modname \
		--loss_name loss_$modname \
		--model_name $modname \
		--test_target yangSMK \
		--test_source louise\
		--resume_from_max 1\
		# --resume_iters 270000 \

