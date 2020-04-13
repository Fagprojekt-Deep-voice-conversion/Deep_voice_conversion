if [ -d "data/VCTK-Data" ] 
then
	echo "Directory exists"
else
	mkdir data/VCTK-Data
	wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
	unzip VCTK-Corpus.zip -d ../data/VCTK-Data
	
	###If the downloaded VCTK is in tar.gz, run this:
	#tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
    
fi
