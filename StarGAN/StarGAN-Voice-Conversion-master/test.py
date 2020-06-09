import os
dir = "/work1/s183920/Deep_voice_conversion/data/VCTK-Data/VCTK-Corpus/wav48"

print([name for name in os.listdir(dir) if os.path.isdir(dir+"/"+name)])

#speakers_used = []
#dir = "/work1/s183920/Deep_voice_conversion/data/VCTK-Data/VCTK-Corpus/wav48"
#for name in os.listdir(dir):
#	if os.path.isdir(dir+"/"+name):
#		print(name)
