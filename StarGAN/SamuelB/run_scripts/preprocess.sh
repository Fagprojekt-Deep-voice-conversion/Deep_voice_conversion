#!/bin/bash

cd ~/*/Deep_voice_conversion/StarGAN/run_scripts

sh setup.sh

module load python3
#module load cuda
source StarGAN-env/bin/activate

# VCTK-Data
python ../StarGAN-Voice-Conversion-master/preprocess.py \
		--resample_rate 16000 \
		--origin_wavpath ../../data/VCTK-Data/VCTK-Corpus/wav48 \
		--target_wavpath ../../data/VCTK-Data/StarGAN/VCTK-Corpus/wav16 \
		--mc_dir_train ../../data/VCTK-Data/StarGAN/mc/train \
		--mc_dir_test ../../data/VCTK-Data/StarGAN/mc/test \
		--speaker_dirs p262 p272 p229 p232 p292 p293 p360 p361 p248 p251
