#!/bin/bash

name=SMK2
origin_folder=SMK_original

# make dirs
cd /work1/s183920/Deep_voice_conversion/data
mkdir $name
mkdir $name/mc
mkdir $name/mc/train
mkdir $name/mc/test
mkdir $name/mc/wav16

# dirs for speakers
mkdir $name/mc/wav16/louise
# mkdir $name/mc/wav16/hilde
mkdir $name/mc/wav16/yangSMK


# copy files - amnually add or change names of test and train speakers

# test speakers
/bin/cp $origin_folder/mc/test/louise* $name/mc/test
# /bin/cp $origin_folder/mc/train/hilde_001.npy $name/mc/test
# /bin/cp $origin_folder/mc/train/hilde_00*.npy $name/mc/test
/bin/cp $origin_folder/mc/train/yangSMK_001.npy $name/mc/test

# train speakers
/bin/cp $origin_folder/mc/train/louise* $name/mc/train
# /bin/cp $origin_folder/mc/train/hilde* $name/mc/train
/bin/cp $origin_folder/mc/train/yangSMK* $name/mc/train

# wav 16
/bin/cp $origin_folder/mc/wav16/louise/* $name/mc/wav16/louise
# /bin/cp $origin_folder/mc/wav16/hilde/* $name/mc/wav16/hilde
/bin/cp $origin_folder/mc/wav16/yangSMK/* $name/mc/wav16/yangSMK