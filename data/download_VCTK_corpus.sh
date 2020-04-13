#!/bin/bash




if [ $PWD = ~/*/Deep_voice_conversion/data ]
then
	if [ -d "VCTK-Data" ] 
	then
		echo -e "\e[32mVCTK corpus already downloaded!\e[0m"
	else
		mkdir VCTK-Data
		wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
		unzip VCTK-Corpus.zip -d VCTK-Data
		
		###If the downloaded VCTK is in tar.gz, run this:
		#tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
	fi
	
else
	echo -e "\e[31mHave you made sure to run this scripts from the 'Deep_voice_conversion/data' directory?\e[0m"

fi
