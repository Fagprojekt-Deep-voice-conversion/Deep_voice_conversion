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
	
	### Load modules
	module load python3
	module load cuda

	### Run setup
	sh setup.sh $run_dir || exit 1

fi

source StarGAN-env/bin/activate

### Convert
python ../StarGAN-Voice-Conversion-master/convert.py \
		--num_speakers 10 \
		--num_converted_wavs 8 \
		--resume_iters 176000 \
		--src_spk p262 \
		--trg_spk p272 \
		--train_data_dir /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/mc/train \
		--test_data_dir /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/mc/test \
		--wav_dir /work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/wav16 \
		--log_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/logs \
		--model_save_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/models \
		--convert_dir /work1/s183920/Deep_voice_conversion/data/results/StarGAN/converted \
