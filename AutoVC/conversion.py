import matplotlib.pyplot as plt
import librosa, time, torch
from dataload import DataLoad2
from Preprocessing_WAV import AutoVC_Mel, WaveRNN_Mel
from Generator_autoVC.model_vc import Generator
from vocoder.WaveNet import build_model
from vocoder.WaveNet import wavegen
from Speaker_encoder.audio import preprocess_wav
from Speaker_encoder.inference import load_model as load_encoder
from Speaker_encoder.inference import embed_utterance
from vocoder.WaveRNN_inference import Generate
from vocoder.WaveRNN_model import WaveRNN
from hparams import hparams_waveRNN as hp

def Conversion(source, target, model,  vocoder = "wavenet", sound_out = False, Visualize = False):

    if vocoder == "wavenet":
        s, t = AutoVC_Mel(source), AutoVC_Mel(target)
    elif vocoder == "wavernn": 
        s, t = WaveRNN_Mel(source), WaveRNN_Mel(target)
    else:
        raise RuntimeError("No vocoder chosen")

    S, T = torch.from_numpy(s.T).unsqueeze(0), torch.from_numpy(t.T).unsqueeze(0)
    
    S_emb, T_emb = embed(source), embed(target)
    
    conversions = {"SS": (S, S_emb, S_emb), "ST": (S, S_emb, T_emb), "TT": (T, T_emb, T_emb), "TS": (T, T_emb, S_emb)}
    converted_numpy = []
    converted_tensor = []
    for key, (X, c_org, c_trg) in conversions.items():
        _, Out, _ = model(X, c_org, c_trg)
        Out1 = Out.squeeze(0).squeeze(0).detach().numpy()
        converted_numpy.append(Out1)
        converted_tensor.append(Out)
    return S.squeeze(0).detach().numpy(), T.squeeze(0).detach().numpy(), tuple(converted_numpy)

def embed(path):
    y = librosa.load(path, sr = 16000)[0]
    y = preprocess_wav(y)
    return torch.tensor(embed_utterance(y)).unsqueeze(0)
    
load_encoder("Models/SpeakerEncoder/SpeakerEncoder.pt").float()

model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/AutoVC/autoVC_full_wavenetaverage_step200k.pt", map_location=torch.device("cpu"))
model.load_state_dict(g_checkpoint['model_state'])
data, labels = DataLoad2("Test_Data")


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




start = time.time()

s = data[len(data)-7]
t = "Peter.wav"

S, T, (SS, ST, TT, TS)  = Conversion(s, t, model, vocoder = "wavernn", Visualize= False, sound_out= False)

A = ["source", "target", "Source_Source", "Source_Target", "Target_Target", "Target_Source"]
B = [S, T, SS, ST, TT, TS]
for i, a in enumerate(A):
    Generate(B[i].T, "ConvertedWavs/" + a + "1" , voc_model)

end = time.time()

print(f"\nTotal time: {end - start}")




