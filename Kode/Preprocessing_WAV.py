#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from resemblyzer.audio import preprocess_wav
class Preproccesing:
    def __init__(self, sr = 16000, n_fft = 1024, hop = 256, n_mels = 40, ref_db = 20):
        self.sampling_rate = sr
        self.n_fft = n_fft
        self.hop_length = hop
        self.n_mels = n_mels
        self.ref_db = ref_db
        self.rescaling_max = 0.999

    def spec_Mel(self, paths, labels):
        """
        Creates Mel Spectrogram from Waveform (.wav).
        Input: .wav filepath
        Output: 80 dim normalised mel-spectrogram
        """
        if type(paths) is not list:
            paths = [paths]
        if type(labels) is not list:
            labels = [labels]
        if len(labels) == 1:
            paths = [paths]
        self.labels = labels
        self.mel_specs = [[] for _ in labels]
        print(len(self.mel_specs))
        for i, label in enumerate(labels):
            print(i, label)
            for path in paths[i]:
                print(path)
                filename = path

                # Load and rescale signal
                y, sr = librosa.load(filename, sr=self.sampling_rate)
                y = preprocess_wav(y)


                # Short Time Fourier Transformation
                STFT = librosa.core.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

                # Convert to power spectrum, and from power to DB using reference power = ref_db
                power = abs(STFT)**2
                S_DB = librosa.power_to_db(power, ref = self.ref_db)

                # Compute Mel-Filterbank with n_mels (default = 80) mels
                mel_basis = librosa.filters.mel(self.sampling_rate, self.n_fft, self.n_mels)

                # Create Mel Spectrogram from STFT and Mel filterbank - Normalise: [0 , 1]
                mel_spectrogram = np.dot(mel_basis, S_DB)
                norm_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
                self.Mel_spectrogram = np.log(norm_mel.T)
                self.mel_specs[i].append(self.Mel_spectrogram)
                print(np.shape(self.Mel_spectrogram))
        return self.mel_specs

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

    def slicer(self, slice_size, mel_list = None, n_mels=None):
        """
        Takes a batch of mel_spectrograms (represented as a list of list),
        slice size and number of mels.
        Computes 50 % overlapping slices of even length (slice_size) of spectrograms and
        concatenates every slice into one 3 d matrix of dimension: a x n_mels x slice_size
        where 'a' depends on the batch size and how the spectrograms are sliced.
        If a spectrogram is not shaped correctly for slicing, zeropadding is performed.

        Inpus Params:
        mel_list: list of list contaning mel_spectrograms
            mel_list[i] = list of spectrograms belonging to person i
            mel_list[i][j] = spectrogram j belonging to person i
        slice_size: (int) the size of each slice (works only for even numbers)
        n_mels: (int) the number of mel channels of the spectrograms

        Output:
        stack_of_slices: 3d matrix of spectrogram slices
        list_of_slices: list of n slices, where n is the number of melspectrograms.
                        slice i indicate where to slice stack_of_slices to get slices
                        corresponding to spectrogram i.

        """
        # Initialize outputs
        if mel_list == None:
            mel_list = self.mel_specs
        if n_mels == None:
            n_mels = self.n_mels

        stack_of_slices = np.empty((1, slice_size,n_mels))
        list_of_slices = []
        old = 0

        # Enumerate over batch
        for i, label in enumerate(mel_list):
            for j, A in enumerate(label):

                xA, yA = np.shape(A)  # Gets shape of spectrogram

                # Perform zero-padding
                diff = xA % int(slice_size / 2)
                to_pad = int((slice_size / 2 - diff) % (slice_size / 2))
                padding = np.zeros((to_pad,yA))
                A_pad = np.vstack((A, padding))

                # Calculate number of slices
                num_slices = int((xA + to_pad) / (slice_size / 2)) - 1
                list_of_slices.append(slice(old, old + num_slices))
                old += num_slices

                # Slice spectrogram and append to mels

                mels = []
                for k in range(int(num_slices)):
                    S = slice(int(k * slice_size / 2), int(k * slice_size / 2 + slice_size))
                    A_slice = A_pad[S]
                    mels.append(A_slice)

                # Stacks slices to matrix and stacks with the rest of sliced spectrograms
                mels = np.stack(mels, axis=0)
                stack_of_slices = np.vstack((stack_of_slices, mels))
            print("slice: ", np.shape(stack_of_slices))
        return stack_of_slices[1:], list_of_slices



