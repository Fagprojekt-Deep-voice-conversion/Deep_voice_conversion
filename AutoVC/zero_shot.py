from conversion import *
import torch
from Preprocessing_WAV import WaveRNN_Mel, AutoVC_Mel
import matplotlib.pyplot as plt
import numpy as np

def Zero_shot(source, target, model, voc_model, save_path, only_conversion = True):
    """
    params:
    source: filepath to source file
    target: filepath to target file
    model: AutoVC model (use Instantiate_Models)
    voc_model: Vocder model (use Instantiate_Models)
    save_path: path to directory to store output
    only_conversion: only outputs converted file. If false source and target are outputted as well
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    s = WaveRNN_Mel(source)
    t = WaveRNN_Mel(target)
   
    S, T = torch.from_numpy(s.T).unsqueeze(0).to(device), torch.from_numpy(t.T).unsqueeze(0).to(device)
    
    S_emb, T_emb = embed(source).to(device), embed(target).to(device)
    
    conversions = {"source": (S, S_emb, S_emb), "Converted": (S, S_emb, T_emb), "target": (T, T_emb, T_emb)}
    

    for key, (X, c_org, c_trg) in conversions.items():
        if key == "Converted":
            _, Out, _ = model(X, c_org, c_trg)
            name = f"{save_path}/{key}"
            print(f"\n Generating {key} sound")
            Generate(Out, name, voc_model)
        else:
            Out = X.unsqueeze(0)
            if not only_conversion:
                name = f"{save_path}/{key}"
                print(f"\n Generating {key} sound")
                Generate(Out, name, voc_model)
            
        
        
        
        
        

# model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/autoVC30min_step72.pt')
if __name__ == "__main__":
    model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/autoVC_seed40_200k.pt')
    Zero_shot("2.wav", "2.wav", model, voc_model, ".")
    # s = "p225_001.wav"
    
    # import librosa
    # import seaborn as sns
    # from hparams import hparams_autoVC as hp
    # from Preprocessing_WAV import *
    # y = load_wav(s)

    # plt.plot(y)
    # plt.ylim(-0.8, 0.8)
    # plt.grid()
    # plt.show()


    # y = y / np.abs(y).max() * hp.rescaling_max
    # y = trim(y)
    # X = abs(librosa.stft(y,
    #                                    n_fft=hp.fft_size,
    #                                    hop_length=hp.hop_size,
    #                                 #    n_mels=hp.num_mels,
    #                                 #    fmin=hp.fmin,
    #                                 #    fmax=hp.fmax,
    #                                    ))**2

    X = spectrogram(y)
    # X = librosa.feature.melspectrogram(y, sr=hp.sample_rate,
    #                                    n_fft=hp.fft_size,
    #                                    hop_length=hp.hop_size,
    #                                    n_mels=hp.num_mels,
    #                                    fmin=hp.fmin,
    #                                    fmax=hp.fmax,
    #                                    power=2,
    #                                    )
    # X = librosa.power_to_db(X, ref=hp.ref_level_db)
    # X = np.clip((X - hp.min_level_db) / (- hp.min_level_db), 0, 1)
    
    plt.matshow(X)
    plt.show()

    plt.plot(X[:,40])
    plt.show()

    # y, sr = librosa.load(s, 16000)
    # Y = librosa.feature.mfcc(y = y, sr = sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc=24)
    # plt.matshow(Y)
    # # plt.plot(y)
    # plt.show()
