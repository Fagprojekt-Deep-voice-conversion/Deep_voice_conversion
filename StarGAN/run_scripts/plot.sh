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


# Plot
python ../plot_loss.py \
