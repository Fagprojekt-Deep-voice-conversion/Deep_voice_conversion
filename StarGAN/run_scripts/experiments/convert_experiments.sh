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
steps=175000
seed=1000
num_wavs=12

anders_files=(3 27 73 120 133 148 167 234 258 279 326 484)
helle_files=(14 98 134 201 243 319 324 345 348 380 428 436)
hillary_files=(10 15 80 97 100 102 114 138 201 215 239 255)
lars_files=(20 48 82 99 171 201 211 269 275 276 297 337)
mette_files=(22 38 46 61 84 90 94 128 151 159 168 180)
michelle_files=(30 52 57 62 130 131 132 135 194 216 233 236)
obama_files=(34 49 79 82 85 107 144 147 187 217 223 231)
trump_files=(21 101 124 127 135 162 180 182 184 238 251)


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
		--files_to_convert "${helle_files[@]}" \
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
		--files_to_convert "${mette_files[@]}" \
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
		--files_to_convert "${lars_files[@]}" \
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
		--files_to_convert "${anders_files[@]}" \
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
		--files_to_convert "${lars_files[@]}" \
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
		--files_to_convert "${anders_files[@]}" \
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
		--files_to_convert "${lars_files[@]}" \
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
		--files_to_convert "${anders_files[@]}" \
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
		--files_to_convert "${helle_files[@]}" \
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
		--files_to_convert "${mette_files[@]}" \
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
		--files_to_convert "${helle_files[@]}" \
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
		--files_to_convert "${mette_files[@]}" \
		--modelstep_dir 0 \

# English Female to Female
echo "Converting English Female to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk michelle \
		--trg_spk hillary \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Female \
		--files_to_convert "${michelle_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk hillary \
		--trg_spk michelle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Female \
		--files_to_convert "${hillary_files[@]}" \
		--modelstep_dir 0 \

# English Male to Male
echo "Converting English Male to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk obama \
		--trg_spk trump \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Male \
		--files_to_convert "${obama_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk trump \
		--trg_spk obama \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Male \
		--files_to_convert "${trump_files[@]}" \
		--modelstep_dir 0 \

# English Male to Female ----
echo "Converting English Male to Female..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk obama \
		--trg_spk hillary \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert "${obama_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk obama \
		--trg_spk michelle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert "${obama_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk trump \
		--trg_spk hillary \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert "${trump_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk trump \
		--trg_spk michelle \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Male_Female \
		--files_to_convert "${trump_files[@]}" \
		--modelstep_dir 0 \

# English Female to Male
echo "Converting English Female to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk hillary \
		--trg_spk obama \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert "${hillary_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk michelle \
		--trg_spk obama \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert "${michelle_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk hillary \
		--trg_spk trump \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert "${hillary_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk michelle \
		--trg_spk trump \
		--train_data_dir $datadir/mc/train \
		--test_data_dir $datadir/mc/test \
		--wav_dir $datadir/wav16 \
		--log_dir $moddir/logs/$modname \
		--model_save_dir $moddir/models/$modname \
		--convert_dir $conv_dir/English/Female_Male \
		--files_to_convert "${michelle_files[@]}" \
		--modelstep_dir 0 \

# Danish Male to Male - 10 min
echo "Converting Danish Male to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk anders \
		--train_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/wav16 \
		--log_dir $moddir/logs/10min \
		--model_save_dir $moddir/models/10min \
		--convert_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/10min/experiment/Male_Male \
		--files_to_convert "${lars_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk lars \
		--train_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/10min/wav16 \
		--log_dir $moddir/logs/10min \
		--model_save_dir $moddir/models/10min \
		--convert_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/10min/experiment/Male_Male \
		--files_to_convert "${anders_files[@]}" \
		--modelstep_dir 0 \

# Danish Male to Male - 20 min
echo "Converting Danish Male to Male..."
python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk lars \
		--trg_spk anders \
		--train_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/wav16 \
		--log_dir $moddir/logs/20min \
		--model_save_dir $moddir/models/20min \
		--convert_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/20min/experiment/Male_Male \
		--files_to_convert "${lars_files[@]}" \
		--modelstep_dir 0 \

python ../StarGAN-Voice-Conversion-master/convert.py \
		--seed $seed \
		--num_converted_wavs $num_wavs \
		--resume_iters $steps \
		--src_spk anders \
		--trg_spk lars \
		--train_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion/data/train_data/StarGAN/20min/wav16 \
		--log_dir $moddir/logs/20min \
		--model_save_dir $moddir/models/20min \
		--convert_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/20min/experiment/Male_Male \
		--files_to_convert "${anders_files[@]}" \
		--modelstep_dir 0 \


