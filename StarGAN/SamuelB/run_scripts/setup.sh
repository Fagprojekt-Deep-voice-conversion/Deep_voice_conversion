#!/bin/bash

cd ~/*/Deep_voice_conversion/StarGAN/run_scripts


### Make python environment
module load python3
python3 -m venv StarGAN-env

source StarGAN-env/bin/activate

python -m pip install torch==1.3.1 torchvision==0.4.2 pillow==5.4.1 pyworld==0.2.8 tqdm==4.41.1 librosa
