import matplotlib.pyplot as plt
import numpy as np
import librosa, time, torch
from dataload import DataLoad2
from Preprocessing_WAV import AutoVC_Mel, WaveRNN_Mel
from Generator_autoVC.model_vc import Generator
from vocoder.WaveNet import build_model
from vocoder.WaveNet import wavegen
from Speaker_encoder.audio import preprocess_wav
from Speaker_encoder.inference import load_model as load_encoder
from Speaker_encoder.inference import embed_utterance
from vocoder.WaveRNN_model import WaveRNN
from hparams import hparams_waveRNN as hp
import pickle
import os

def Instantiate_Models(model_path,  vocoder = "wavernn", sound_out = False, Visualize = False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Prepare vocoder
    if vocoder == "wavenet":
        
        # Instantiate WaveNet Model
        voc_model = build_model().to(device)
        checkpoint = torch.load("Models/WaveNet/WaveNetVC_pretrained.pth", map_location=torch.device(device))
        voc_model.load_state_dict(checkpoint["state_dict"])

    elif vocoder == "wavernn": 
        
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
                            mode='MOL').to("cpu")

        voc_model.load('Models/WaveRNN/WaveRNN_Pretrained.pyt')
    else:
        raise RuntimeError("No vocoder chosen")

    # Prepare AutoVC model
    model = Generator(32, 256, 512, 32).eval().to(device)
    g_checkpoint = torch.load("Models/AutoVC/" + model_path + ".pt", map_location=torch.device(device))
    model.load_state_dict(g_checkpoint['model_state'])

    # Prepare Speaker Encoder Module
    load_encoder("Models/SpeakerEncoder/SpeakerEncoder.pt").float()
        
    return model, voc_model

def embed(path):
    y = librosa.load(path, sr = 16000)[0]
    y = preprocess_wav(y)
    return torch.tensor(embed_utterance(y)).unsqueeze(0)

def Generate(m, fpath, model, modeltype = "wavernn"):
    
    if modeltype == "wavernn":
        m = m.squeeze(0).squeeze(0).T.unsqueeze(0)
        waveform = model.generate(m, batched = True, target = 11_000, overlap = 550, mu_law= False)

    elif modeltype == "wavenet":
        m = m.squeeze(0).squeeze(0)
        waveform = wavegen(model, m)

    librosa.output.write_wav(fpath + '.wav', np.asarray(waveform), sr = hp.sample_rate)




def Conversion(source, target, model, voc_model, voc_type = "wavernn", task = None, subtask = None):
    if voc_type == "wavernn":
        s, t = WaveRNN_Mel(source), WaveRNN_Mel(target)
    else:
        s, t = AutoVC_Mel(source), AutoVC_Mel(target)

    S, T = torch.from_numpy(s.T).unsqueeze(0), torch.from_numpy(t.T).unsqueeze(0)
    
    S_emb, T_emb = embed(source), embed(target)
    
    conversions = {"Source": (S, S_emb, S_emb), "Converted": (S, S_emb, T_emb), "Target": (T, T_emb, T_emb)}
    try:
        dir_size = len(list(os.walk(f"Experiments/AutoVC/{task}/{subtask}"))[0][1]) + 1
    except:
        dir_size = 1

    os.mkdir(f"Experiments/AutoVC/{task}/{subtask}/{dir_size}")

    for key, (X, c_org, c_trg) in conversions.items():
        if key == "Converted":
            _, Out, _ = model(X, c_org, c_trg)
        else:
            Out = X.unsqueeze(0)
        
        
        path = f"Experiments/AutoVC/{task}/{subtask}/{dir_size}/{key}"
        print(f"\n Generating {key} sound")
        Generate(Out, path, voc_model, voc_type)


if __name__ == "__main__":
    data, labels = DataLoad2("Test_Data")

    s = data[0]
    t = data[-1]
    model, voc_model = Instantiate_Models(model_path = "autoVC_seed20_200k", vocoder = "wavernn", Visualize= False, sound_out= False)

    Conversion(s, t, model, voc_model, task = "English_English", subtask = "Male_Male")
    
    # X = pickle.load(open("Models/loss_scratch40", "rb"))
    # Y = pickle.load(open("Models/loss_scratch60", "rb"))
    # Z = pickle.load(open("Models/loss_transfer20", "rb"))


    # plt.plot(X); plt.plot(Y); plt.plot(Z)
    # plt.grid()
    # plt.show()
