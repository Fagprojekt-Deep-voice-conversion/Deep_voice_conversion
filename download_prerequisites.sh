### Download git repo
if [ -d "../Deep_voice_conversion" ]
then 
	echo "Repo already downloaded"
else
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
	echo "StarGAN model repo already downloaded"
else
	wget https://github.com/SamuelBroughton/StarGAN-Voice-Conversion/archive/master.zip
	unzip master.zip -d ./
	rm master.zip


fi




