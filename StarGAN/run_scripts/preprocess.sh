#!/bin/bash
### Set directory

BASE_DIRECTORY=$(echo "$PWD" | cut -d "/" -f2)
if [ $BASE_DIRECTORY = "work1" ]
then 
	run_dir=/work1/s183920/Deep_voice_conversion/StarGAN/run_scripts
else
	run_dir=~/Desktop/Deep_voice_conversion/StarGAN/run_scripts/
fi




### Load modules
module load python3
module load cuda

### Run setup
sh setup.sh $run_dir || exit 1
source StarGAN-env/bin/activate

### Preprocess
### Speakers must be set in the python script
python ../StarGAN-Voice-Conversion-master/preprocess.py \
		--sample_rate 16000 \
		--origin_wavpath /work1/s183920/Deep_voice_conversion/data/VCTK-Data/VCTK-Corpus/wav48 \
		--target_wavpath /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/wav16 \
		--mc_dir_train /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/mc/train \
		--mc_dir_test /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/mc/test
