#!/bin/bash

cd ~/*/Deep_voice_conversion/Model/run_scripts


### Make python environment
module load python3
python3 -m venv AutoVC-env

source AutoVC-env/bin/activate

python -m pip install torch==1.4.0 sklearn tqdm librosa torchvision webrtcvad

deactivate


