import torch
from WaveRNN_model import WaveRNN
from AutoVC.hparams import hparams_waveRNN as hp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os,sys; os.chdir(sys.path[0])
from AutoVC.Preprocessing_WAV import WaveRNN_Mel


def Generate(m):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode='MOL').to(device)

    voc_model.load('utils/WaveRNN_Pretrained.pyt')



 
    m = torch.tensor(m).unsqueeze(0)
    print(m.shape)
    # m = (m + 4) / 8

    waveform = voc_model.generate(m, batched = False, target = 11_000, overlap = 550, mu_law= False)
    plt.plot(waveform)
    plt.show()

    librosa.output.write_wav("test1" + '.wav', np.asarray(waveform), sr = hp.sample_rate)

M = WaveRNN_Mel("p225_001.wav")


Generate(M)