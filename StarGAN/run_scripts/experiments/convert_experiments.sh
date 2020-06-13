#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Conversion
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set span if number of cores is more than 1
###BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request xGB of system-memory
#BSUB -R "rusage[mem=30GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Conversion_%J.out
#BSUB -e Conversion_%J.err
# -- end of LSF options --

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

datadir=/work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/30min
moddir=/work1/s183920/Deep_voice_conversion/StarGAN
modname=30min_seed1000
conv_dir=/work1/s183920/Deep_voice_conversion/data/results/StarGAN/$modname/experiment 
steps=190000
seed=1000
num_wavs=1

lars_files=random
anders_files=random
helle_files=134
mette_files=(180 225)


### Load modules
module load python3
source StarGAN-env/bin/activate

### Convert
### Must set the speakers that was used for preprocessing in script

# Danish Female to Female
echo "Converting Danish Female to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Female \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Female \
		--files_to_convert $mette_files \
		--modelstep_dir 0 \

# Danish Male to Male
echo "Converting Danish Male to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Male \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Male \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

# Danish Male to Female
echo "Converting Danish Male to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Female \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Female \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Female \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Male_Female \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

# Danish Female to Male
echo "Converting Danish Female to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Male \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Male \
		--files_to_convert $mette_files\
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Male \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/Danish/Female_Male \
		--files_to_convert $mette_files \
		--modelstep_dir 0 \

# English Female to Female
echo "Converting English Female to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Female \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Female \
		--files_to_convert $mette_files \
		--modelstep_dir 0 \

# English Male to Male
echo "Converting English Male to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Male \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Male \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

# English Male to Female
echo "Converting English Male to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk helle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert $lars_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk mette \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert $anders_files \
		--modelstep_dir 0 \

# English Female to Male
echo "Converting English Female to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk lars \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert $mette_files\
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk helle \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert $helle_files \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk mette \
		--trg_spk anders \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert $mette_files \
		--modelstep_dir 0 \

