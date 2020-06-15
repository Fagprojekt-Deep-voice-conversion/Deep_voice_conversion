#!/usr/bin/env python3

import argparse
import csv
import multiprocessing
import shutil
import os

import subprocess

# if __name__ == "__main":
err = subprocess.Popen(['pip', 'install', 'youtube-dl'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE).communicate()[1]

if len(err) != 0:
    print(err)
    exit(1)

def download(name, link, outfile):
    import subprocess
    
    print('Downloading', name, 'from', link + '...')
    process = subprocess.Popen(['youtube-dl',
                                '--extract-audio',
                                '--audio-format', 'wav',
                                '-o', outfile, link],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    
    _, err = process.communicate()
    
    if err != b'':
        import sys
        print('Error downloading', name, 'from', link, file=sys.stderr)
        print(err.decode('utf-8'), file=sys.stderr)

def download_all(dl_path = "tracks", csv_name = "sange.csv"):#, sound_type = "tracks"):
    """
    Downloades all links from the csv
    ------------------------------------
    csv_name = the csv to download from, the first column must be the name of the track and the second must be a link \n
    dl_path = the path in which to place the downloaded tracks \n
    sound_type = whick type of sound, a folder with this name will be created with the downloads if no dl_path is specified
    """
    # tracks_path = os.path.join(os.path.curdir, sound_type) if dl_path is None else dl_path
    tracks_path = dl_path
    name_index = 0
    link_index = 1
    
    if os.path.exists(tracks_path):
        shutil.rmtree(tracks_path)
    
    os.mkdir(tracks_path)
    
    with multiprocessing.Pool() as p:
        with open(csv_name, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            
            for i, row in enumerate(reader, 1):
                name = row[name_index]
                link = row[link_index]
                
                outfile = os.path.join(tracks_path, str(i) + '.wav')
                p.apply_async(download, (name, link, outfile))
        
        p.close()
        p.join()