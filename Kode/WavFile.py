import os
import pydub
from pathlib import Path
from pydub import AudioSegment

AudioSegment.converter = r"C:\\ffmpeg\\bin\\ffmpeg.exe"
for dir in os.listdir("Speakers"):
    print(dir)
    for file in os.listdir("Speakers/{}".format(dir)):
        t1 = 0
        t2 = 5000
        print(file)
        speaker = file.split(".")[0]
        filename = 0
        sound = AudioSegment.from_mp3("Speakers/{}/{}".format(dir,file))
        os.mkdir("fiveSecondFiles/{}".format(speaker))
        while t2 < len(sound):
            print("exporting {} {}...".format(speaker, filename))
            newAudio = sound[t1:t2]
            filename += 1
            newAudio.export("fiveSecondFiles/{}/{}_{}".format(speaker, speaker, filename), format="wav")
            t1 += 5000
            t2 += 5000

'''
audio_chunks = pydub.silence.split_on_silence(file, 
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)'''