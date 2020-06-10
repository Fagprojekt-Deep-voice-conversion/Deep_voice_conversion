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
module load python3
module load cuda

source StarGAN-env/bin/activate

### Train model
#python ../StarGAN-Voice-Conversion-master/main.py

### Model config \
### Training config \
### Test config \
### Miscellaneous \
### Step size \
### Directories \

python ../StarGAN-Voice-Conversion-master/main.py \
		--lambda_cls 10 \
		--lambda_rec 10 \
		--lambda_gp 10 \
		--sampling_rate 16000 \
		--batch_size 32 \
		--num_iters 8000 \
		--num_iters_decay 100000 \
		--g_lr 0.0001 \
		--d_lr 0.0001 \
		--n_critic 5 \
		--beta1 0.5 \
		--beta2 0.999 \
		--test_iters 100000 \
		--seed 420 \
		--num_workers 1 \
		--mode train \
		--log_step 10 \
		--sample_step 1000 \
		--model_save_step 1000 \
		--lr_update_step 1000 \
		--train_data_dir /work1/s183920/Deep_voice_conversion_old/data/VCTK-Data/StarGAN/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion_old/data/VCTK-Data/StarGAN/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion_old/data/VCTK-Data/StarGAN/wav16 \
		--log_dir /work1/s183920/Deep_voice_conversion_old/data/results/StarGAN/logs \
		--sample_dir /work1/s183920/Deep_voice_conversion_old/data/results/StarGAN/samples \
		--model_save_dir /work1/s183920/Deep_voice_conversion_old/data/results/StarGAN/models \
		--loss_name loss_test \
		--resume_iters 4000\

