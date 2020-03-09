import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def spec_Mel(path):
    #Returns the spectogram matrix of a desired filename
    filename = path
    y, sr = librosa.load(filename)
    # trim silent edges
    Test, _ = librosa.effects.trim(y)

    #librosa.display.waveplot(Test, sr=sr);

    print("Sampling Rate: {}".format(sr))
    n_fft = 2048 #frame length. The (positive integer) number of samples in an analysis window (or frame). This is denoted by an integer variable n_fft.
    hop_length = 512 #The number of samples between successive frames, e.g., the columns of a spectrogram. This is denoted as a positive integer hop_length.
    n_mels = 128 #the non linear transformation value to transform HZ to mel's
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)


    S = librosa.feature.melspectrogram(Test, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max) #Shape: 128 X 203
    print('Shape of the matrix: {}'.format(np.shape(S)))
    return S, S_DB