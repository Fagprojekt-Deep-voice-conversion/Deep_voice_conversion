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
# module load python3
module load cuda

### Run setup
sh setup.sh $run_dir || exit 1
source StarGAN-env/bin/activate

### Preprocess
### Submit must be set to CPU and mulitple cores can be requested
### Speakers must be set in the python script
data_subfolder=SMK4
python ../StarGAN-Voice-Conversion-master/preprocess.py \
		--sample_rate 16000 \
		--origin_wavpath /work1/s183920/Deep_voice_conversion/data/$data_subfolder/wavs \
		--target_wavpath /work1/s183920/Deep_voice_conversion/data/$data_subfolder/wav16 \
		--mc_dir_train /work1/s183920/Deep_voice_conversion/data/$data_subfolder/mc/train \
		--mc_dir_test /work1/s183920/Deep_voice_conversion/data/$data_subfolder/mc/test \
		--test_size 0.0001 \
		--resample 1 \
		--prep_train 1 \
		--prep_test 1 \
		--overwrite_old 1 \
		# --train_size 360 \
