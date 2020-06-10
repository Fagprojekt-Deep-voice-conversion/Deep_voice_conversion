import os
import re
dir = "/work1/s183920/Deep_voice_conversion_old/data/VCTK-Data/StarGAN/mc/test"

trg = "p262"

#s = re.search(".*(/Deep.*)/data", dir)
#print("----------------", s.group(1), "--------------------------")

speakers = []

for file in os.listdir(dir):
	s = re.search("(.*)_.*", file).group(1)
	if s not in speakers:
		speakers.append(s)
		
print(speakers)
	

#s = re.search("(.*)_.*",os.listdir(dir))
#print(os.path.exists(dir+"/"+trg+"_.*"))

#speakers_used = []
#dir = "/work1/s183920/Deep_voice_conversion/data/VCTK-Data/VCTK-Corpus/wav48"
#for name in os.listdir(dir):
#	if os.path.isdir(dir+"/"+name):
#		print(name)
