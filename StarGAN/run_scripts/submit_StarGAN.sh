#!/bin/sh
### General options

###  specify queue --
###BSUB -q gpuv100
#BSUB -q hpc

### -- set the job Name --
#BSUB -J StarGAN

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00

### request xGB of system-memory
#BSUB -R "rusage[mem=20GB]"

### -- set the email address --
##BSUB -u s183920@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo StarGAN_%J_output.out
#BSUB -eo StarGAN_%J_error.err

# -- end of LSF options --


sh preprocess.sh
sh train_model.sh
sh convert.sh
