#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class Preproccesing:
    def __init__(self, sr = 16000, n_fft = 1024, hop = 256, n_mels = 80, ref_db = 20):
        self.sampling_rate = sr
        self.n_fft = n_fft
        self.hop_length = hop
        self.n_mels = n_mels
        self.ref_db = ref_db
        self.rescaling_max = 0.999

    def spec_Mel(self, path):
        """
        Creates Mel Spectrogram from Waveform (.wav).
        Input: .wav filepath
        Output: 80 dim normalised mel-spectrogram
        """
        filename = path

        # Load and rescale signal
        y, sr = librosa.load(filename, sr=self.sampling_rate)
        rescaled_signal = y / np.abs(y).max() * self.rescaling_max

        # Short Time Fourier Transformation
        STFT = librosa.core.stft(rescaled_signal, n_fft=self.n_fft, hop_length=self.hop_length)

        # Convert to power spectrum, and from power to DB using reference power = ref_db
        power = abs(STFT)**2
        S_DB = librosa.power_to_db(power, ref = self.ref_db)

        # Compute Mel-Filterbank with n_mels (default = 80) mels
        mel_basis = librosa.filters.mel(self.sampling_rate, self.n_fft, self.n_mels)

        # Create Mel Spectrogram from STFT and Mel filterbank - Normalise: [0 , 1]
        mel_spectrogram = np.dot(mel_basis, S_DB)
        norm_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
        self.Mel_spectrogram = norm_mel

        return self.Mel_spectrogram

    def ShowSpec(self, librosa = False):
        """
        Shows mel spectrogram
        if librosa - plots librosas mel option.
        """
        if librosa:
            librosa.display.specshow(self.Mel_spectrogram, sr = self.sampling_rate, hop_length = self.hop_length,
                                 y_axis = 'fft', x_axis = 's')
        else:
            plt.matshow(self.Mel_spectrogram)
            plt.colorbar()
        plt.show()

    def TrimSilence(self):
        TODO



