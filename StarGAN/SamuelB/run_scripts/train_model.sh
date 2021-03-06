#!/bin/bash

cd ~/*/Deep_voice_conversion/StarGAN/SamuelB/run_scripts

sh setup.sh

module load python3
#module load cuda
source StarGAN-env/bin/activate

### Train model
python ../StarGAN-Voice-Conversion-master/main.py \
		--train_data_dir ../../../data/VCTK-Data/StarGAN/SamuelB/mc/train \
		--test_data_dir ../../../data/VCTK-Data/StarGAN/SamuelB/mc/test \
		--use_tensorboard False \
		--wav_dir ../../../data/VCTK-Data/StarGAN/SamuelB/VCTK-Corpus/wav16 \
		--model_save_dir ../models \
		--sample_dir ../../../data/VCTK-Data/StarGAN/SamuelB/samples \
		--num_iters 200000 \
		--batch_size 8 \
		--speakers p262 p272 p229 p232 \
		--num_speakers 4
