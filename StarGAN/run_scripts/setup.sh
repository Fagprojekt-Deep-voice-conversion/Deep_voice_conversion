#!/bin/bash

### Set directory

if [ "$1" = "" ]
then 
	echo -e "\e[31mMissing a directory to do the setup in\e[0m"
	exit 1
else
	echo "$1"	
	cd $1
fi

if [ ! -d "$1" ] 
then
	echo -e "\e[31mSetup directory does not exist!\e[0m"
	exit 1
fi



### Make python environment
module load python3
python3 -m venv StarGAN-env

source StarGAN-env/bin/activate

python -m pip install --upgrade cython
python -m pip install torch==1.4.0 pyworld tqdm librosa tensorboardX tensorboard torchvision==0.5.0 matplotlib

#sudo apt-get install sox

deactivate
