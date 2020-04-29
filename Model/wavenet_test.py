# import os, sys
# sys.path.append(os.path.abspath(os.curdir))

from Kode.Preprocessing_WAV import Preproccesing
from Kode.dataload import DataLoad2
import numpy as np
import matplotlib.pyplot as plt
import librosa

path = "p227_003.wav"


def play(wav_path):
    import pyaudio  
    import wave  

    #define stream chunk   
    chunk = 1024  

    #open a wav format music  
    f = wave.open(wav_path,"rb")  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate() 

def get_mel(wav_path):
    Prep = Preproccesing()
    Mels, uncorrupted, corrupted = Prep.Mel_Batch(path)
    Mels = Mels[0].cpu()
    Mels = Mels.squeeze()
    return Mels.numpy()

def show_mel(mel_spec):
    plt.matshow(mel_spec)
    plt.show()

mel = get_mel(path)
#show_mel(mel)

#mel, sr = librosa.load(path)
#print(sr)
#mel = librosa.feature.melspectrogram(y=mel, sr=16000)

sound = librosa.feature.inverse.mel_to_audio(mel, sr=16000, n_fft=1024, hop_length=256, win_length=None, 
window='hann', center=True, pad_mode='reflect', power=1.0, n_iter=32, length=None)

librosa.output.write_wav("test.wav", sound, sr = 16000, norm=False)
play("test.wav")