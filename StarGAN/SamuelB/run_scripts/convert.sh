#!/bin/bash

cd ~/*/Deep_voice_conversion/StarGAN/SamuelB/run_scripts

sh setup.sh

module load python3
#module load cuda
source StarGAN-env/bin/activate

### Convert
python ../StarGAN-Voice-Conversion-master/convert.py \
		--resume_model 120000 \
		--num_speakers 4 \
		--speakers p262 p272 p229 p232 \
		--train_data_dir ../../../data/VCTK-Data/StarGAN/SamuelB/mc/train / \
		--test_data_dir ../../../data/VCTK-Data/StarGAN/SamuelB/mc/test/ \
		--wav_dir ../../../data/VCTK-Data/StarGAN/SamuelB/VCTK-Corpus/wav16 \
		--model_save_dir ../models \
		--convert_dir ../../../data/VCTK-Data/StarGAN/SamuelB/converted
