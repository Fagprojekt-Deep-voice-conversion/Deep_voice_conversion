#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import numpy as np
from tqdm import tqdm
from hparams import hparams_autoVC as hp
from hparams import hparams_waveRNN as hp1
import torch
from scipy.io import wavfile

def Mel_Batch(Batch, vocoder = "autovc"):
    if type(Batch) is not list:
        Batch = [Batch]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Mels = []
    uncorrupted = np.ones(len(Batch), dtype = np.bool)
    print("Creating Mel Spectrograms...")

    for i, wav in tqdm(enumerate(Batch)):
        try:
            if vocoder == "autovc":
                X = AutoVC_Mel(wav)
            elif vocoder == "wavernn":
                X = WaveRNN_Mel(wav)
            else:
                raise RuntimeError("Unknown vocoder")
            X = torch.from_numpy(X.T).to(device).unsqueeze(0)
            Mels.append(X)
        except:
            print("Issue with Wav file")
            uncorrupted[i] = False
    return Mels, uncorrupted


def AutoVC_Mel(path):

    y = load_wav(path)
    y = y / np.abs(y).max() * hp.rescaling_max
    y = trim(y)
    X = librosa.feature.melspectrogram(y, sr=hp.sample_rate,
                                       n_fft=hp.fft_size,
                                       hop_length=hp.hop_size,
                                       n_mels=hp.num_mels,
                                       fmin=hp.fmin,
                                       fmax=hp.fmax,
                                       power=hp.power,
                                       )
    X = librosa.amplitude_to_db(X, ref=hp.ref_level_db)
    X = np.clip((X - hp.min_level_db) / (- hp.min_level_db), 0, 1)
    return X

def WaveRNN_Mel(path):
    y = librosa.load(path, sr=hp1.sample_rate)[0]
    X = melspectrogram(y)
    return X

def trim(quantized):
    start, end = start_and_end_indices(quantized, hp.silence_threshold)
    return quantized[start:end]

def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    if sr != hp.sample_rate:
        x = librosa.resample(x, sr, hp.sample_rate)
    x = np.clip(x, -1.0, 1.0)
    return x

def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp1.sample_rate, n_fft=hp1.n_fft, n_mels=hp1.num_mels, fmin=hp1.fmin)

def normalize(S):
    return np.clip((S - hp1.min_level_db) / -hp1.min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -hp1.min_level_db) + hp1.min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp1.ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp1.n_fft, hop_length=hp1.hop_length, win_length=hp1.win_length)









