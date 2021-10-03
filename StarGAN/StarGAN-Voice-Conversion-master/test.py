import os
import re

"""
speak = {}
dir = "/work1/s183920/Deep_voice_conversion/StarGAN/models/base"
if len(os.listdir(dir)) == 0:
	resume_iters = None
else:
	list_of_files = os.listdir(dir)
	def extract_number(f):
		s = re.findall("\d+$",f)
		return (int(s[0]) if s else -1,f)

	resume_iters = int(re.search("(\d*)-.\.ckpt", max(list_of_files,key=extract_number)).group(1))
print(resume_iters)
"""
"""	
	
	if s not in speak.keys():
		speak[str(s)] = 1
	else:
		speak[str(s)] += 1

print(sorted([s for s in speak.keys() if speak[s] <= 10]))
"""
dir = "/work1/s183920/Deep_voice_conversion/StarGAN/models/30min_seed1000/"

nums = [int(re.search("(\d*)-.\.ckpt", file).group(1)) for file in os.listdir(dir)]
print(nums)
print(f"The max is {max(nums)}")

"""
list_of_files = os.listdir(dir)
def extract_number(f):
	s = re.findall("\d+$",f)
	return (int(s[0]) if s else -1,f)
print(sorted(list_of_files))
print(max(list_of_files,key=extract_number))
resume_iters = int(int(re.search("(\d*)-.\.ckpt", max(list_of_files,key=extract_number)).group(1))-1000)
resume_iters = resume_iters if resume_iters >= 1000 else None
print(f"Resuming from iteration {resume_iters}...")
"""
