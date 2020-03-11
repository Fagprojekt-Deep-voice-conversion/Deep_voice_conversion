#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def spec_Mel(path):
    #Returns the spectogram matrix of a desired filename
    filename = path
    y, sr = librosa.load(filename)
    # trim silent edges
    Test, _ = librosa.effects.trim(y, top_db = 1000)

    #librosa.display.waveplot(Test, sr=sr);

    print("Sampling Rate: {}".format(sr))
    n_fft = 1024 #frame length. The (positive integer) number of samples in an analysis window (or frame). This is denoted by an integer variable n_fft.
    hop_length = 256  #The number of samples between successive frames, e.g., the columns of a spectrogram. This is denoted as a positive integer hop_length.
    n_mels = 80 #the non linear transformation value to transform HZ to mel's
    sample_rate = sr
    fmin = 40
    fmax = 7900
    mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)


    S = librosa.feature.melspectrogram(Test, sr=sample_rate, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels, fmin = fmin, fmax = fmax)
    S_DB = librosa.power_to_db(S, ref=np.max) #Shape: 128 X 203

    print('Shape of the matrix: {}'.format(np.shape(S)))
    return S, S_DB



def plot_Mel(path):
    #creates a Mel spectogram plot of the desired filename, with path being a string.
    filename = path
    y, sr = librosa.load(filename)
    # trim silent edges
    Test, _ = librosa.effects.trim(y)
    librosa.display.waveplot(Test, sr=sr);

    n_fft = 1024  # frame length. The (positive integer) number of samples in an analysis window (or frame). This is denoted by an integer variable n_fft.
    hop_length = 256  # The number of samples between successive frames, e.g., the columns of a spectrogram. This is denoted as a positive integer hop_length.
    n_mels = 80  # the non linear transformation value to transform HZ to mel's
    fmin = 125
    fmax = 7600
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    S = librosa.feature.melspectrogram(Test, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels, fmin = fmin, fmax
                                        = fmax)
    S_DB = librosa.power_to_db(S, ref=np.max)  # Shape: 128 X 203
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB');

def play_Spec(spectrogram):
    #spectrogram, _ = spectrogram
    Sound = librosa.feature.inverse.mel_to_stft(spectrogram)
    y = librosa.griffinlim(Sound)
    return plt.plot(y)
