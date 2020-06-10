import os
import re


speak = {}
dir = "/work1/s183920/Deep_voice_conversion/data/VCTK-Data/StarGAN/mc/test"
for f in os.listdir(dir):
	s = re.search("(.*)_.*", f).group(1)
	if s not in speak.keys():
		speak[str(s)] = 1
	else:
		speak[str(s)] += 1

print(sorted([s for s in speak.keys() if speak[s] <= 10]))
