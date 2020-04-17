#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Speaker_encoder.audio import preprocess_wav
import torch

class Preproccesing:
    def __init__(self, sr = 16000, n_fft = 1024, hop = 256, n_mels = 80, ref_db = 20,
                 fmax = 7600, fmin = 125, min_db = -100, rescaling_max = 0.99, power = 1):
        self.sampling_rate, self.n_fft, self.hop_length = sr, n_fft, hop
        self.n_mels, self.ref_db, self.min_db = n_mels, ref_db, min_db
        self.fmax, self.fmin, self.rescaling_max = fmax, fmin, rescaling_max
        self.power = power



    def ShowSpec(self, librosa = False):
        """
        Shows mel spectrogram
        if librosa - plots librosas mel option.
        """

        for i,label in enumerate(self.mel_specs):
            for spec in label:
                if librosa:
                    librosa.display.specshow(spec, sr = self.sampling_rate, hop_length = self.hop_length,
                                         y_axis = 'fft', x_axis = 's')
                else:
                    plt.matshow(spec)
                    plt.colorbar()
            plt.title(self.labels[i])
            plt.show()

    def Mel_Batch(self, Batch):
        if type(Batch) is not list:
            Batch = [Batch]
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        Mels = []
        print("Creating Mel Spectrograms...")
        for wav in tqdm(Batch):
            y, _ = librosa.load(wav, sr = self.sampling_rate)
            y = preprocess_wav(y)
            y = y / np.abs(y).max() * self.rescaling_max

            X = librosa.feature.melspectrogram(y, sr = self.sampling_rate,
                                               n_fft = self.n_fft,
                                               hop_length = self.hop_length,
                                               n_mels = self.n_mels,
                                               fmin = self.fmin,
                                               fmax = self.fmax,
                                               power = self.power,
                                               )
            X = librosa.amplitude_to_db(X, ref = self.ref_db)
            X = np.clip((X - self.min_db) / - self.min_db, 0 ,1)

            X = torch.from_numpy(X.T).to(device).unsqueeze(0)
            Mels.append(X)
        return Mels



