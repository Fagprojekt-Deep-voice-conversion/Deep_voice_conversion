#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J StarGAN_30min_seed2000
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set span if number of cores is more than 1
###BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
# request gpu with 32 gb
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request xGB of system-memory
#BSUB -R "rusage[mem=15GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### Make job dependent on previous
#BSUB -w "ended(7118527)"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o StarGAN_30min_seed2000_%J.out
#BSUB -e StarGAN_30min_seed2000_%J.err
# -- end of LSF options --

### Set directory

if [ "$1" = "setup" ]
then
	BASE_DIRECTORY=$(echo "$PWD" | cut -d "/" -f2)
	if [ $BASE_DIRECTORY = "work1" ]
	then 
		run_dir=/work1/s183920/Deep_voice_conversion/StarGAN/run_scripts
	else
		run_dir=~/Desktop/Deep_voice_conversion/StarGAN/run_scripts/
	fi
	

	### Run setup
	sh setup.sh $run_dir || exit 1

fi

cd ~/Desktop/Deep_voice_conversion/StarGAN/run_scripts/

### Load modules
module load python3
module load cuda

#sh setup.sh $run_dir || exit 1
source StarGAN-env/bin/activate

### Train model
#python ../StarGAN-Voice-Conversion-master/main.py
datadir=/work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/30min
moddir=/work1/s183920/Deep_voice_conversion/StarGAN
modname=30min_seed2000
steps=200000

if [ ! -d "$moddir/models/$modname" ]
then
	mkdir $moddir/models/$modname
fi


python ../StarGAN-Voice-Conversion-master/main.py \
		--lambda_cls 10 \
		--lambda_rec 10 \
		--lambda_gp 10 \
		--sampling_rate 16000 \
		--batch_size 8 \
		--num_iters $steps \
		--num_iters_decay 100000 \
		--g_lr 0.0001 \
		--d_lr 0.0001 \
		--n_critic 5 \
		--beta1 0.5 \
		--beta2 0.999 \
		--test_iters 100000 \
		--seed 2000 \
		--num_workers 1 \
		--mode train \
		--log_step 10 \
		--sample_step 1000 \
		--model_save_step 1000 \
		--lr_update_step 1000 \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--loader_dir $datadir/mc/loader \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--sample_dir $moddir/samples/$modname \
		--model_save_dir $moddir/models/$modname \
		--loss_name loss_$modname \
		--resume_from_max 1 \

