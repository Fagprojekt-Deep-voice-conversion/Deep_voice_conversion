#!/bin/bash

cd ~/*/Deep_voice_conversion/StarGAN/run_scripts

sh setup.sh

module load python3
#module load cuda
source StarGAN-env/bin/activate

### Train model
python ../StarGAN-Voice-Conversion-master/main.py \
		--train_data_dir ../../data/VCTK-Data/StarGAN/mc/train \
		--test_data_dir ../../data/VCTK-Data/StarGAN/mc/test \
		--use_tensorboard False \
		--wav_dir ../../data/VCTK-Data/StarGAN/VCTK-Corpus/wav16 \
		--model_save_dir ../../data/VCTK-Data/StarGAN/models \
		--sample_dir ../../data/VCTK-Data/StarGAN/samples \
		--num_iters 200000 \
		--batch_size 8 \
		--speakers p262 p272 p229 p232 \
		--num_speakers 4
