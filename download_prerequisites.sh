### Download git repo
if [ -d "../Deep_voice_conversion" ]
then 
	echo -e "\e[32mDeep_voice_conversion repo already downloaded\e[0m"
else
	echo -e "\e[93mCloning Deep_voice_conversion repo...\e[0m"
	git clone https://github.com/Fagprojekt-Deep-voice-conversion/Deep_voice_conversion.git
	mv download_prerequisites.sh Deep_voice_conversion
	cd Deep_voice_conversion
fi

### Switch to StarGAN branch
git checkout -t origin/StarGAN

if [ -d "StarGAN" ]
then
	cd StarGAN
else
	mkdir StarGAN
	cd StarGAN
fi

### Download StarGAN repo from Samuel Broughton
if [ -d "StarGAN-Voice-Conversion-master" ]
then 
	echo -e "\e[32mStarGAN model repo already downloaded\e[0m"
else
	echo -e "\e[93mDownloading StarGAN repo from Samuel Broghton\e[0m"
	wget https://github.com/SamuelBroughton/StarGAN-Voice-Conversion/archive/master.zip
	unzip master.zip -d ./
	rm master.zip


fi

### Download VCTK corpus
if [ -d "../data/VCTK-Data/VCTK-Corpus" ] 
then
	echo -e "\e[32mVCTK corpus already downloaded\e[0m"
else
	echo -e "\e[31mPlease run 'sh data/download_VCTK_corpus.sh'\e[0m"
    
fi



