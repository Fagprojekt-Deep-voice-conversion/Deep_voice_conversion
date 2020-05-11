import matplotlib.pyplot as plt
from Generator_autoVC.model_vc import Generator
import torch
from dataload import DataLoad2
from Preprocessing_WAV import AutoVC_Mel, WaveRNN_Mel
from Speaker_identity import SpeakerIdentity
from vocoder.WaveNet import build_model
from vocoder.WaveNet import wavegen
import librosa
from vocoder.WaveRNN_inference import Generate
from vocoder.WaveRNN_model import WaveRNN
import time

from hparams import hparams_waveRNN as hp
def Conversion(source, target, model,  vocoder = "wavenet", sound_out = False, Visualize = False):

    if vocoder == "wavenet":
        s, t = AutoVC_Mel(source), AutoVC_Mel(target)
    elif vocoder == "wavernn": 
        s, t = WaveRNN_Mel(source), WaveRNN_Mel(target)
    else:
        raise RuntimeError("No vocoder chosen")

    S, T = torch.from_numpy(s.T).unsqueeze(0), torch.from_numpy(t.T).unsqueeze(0)
    
    S_emb, T_emb = SpeakerIdentity(source)[0], SpeakerIdentity(target)[0]
    
    conversions = {"SS": (S, S_emb, S_emb), "ST": (S, S_emb, T_emb), "TT": (T, T_emb, T_emb), "TS": (T, T_emb, S_emb)}

    converted_numpy = []
    converted_tensor = []
    for key, (X, c_org, c_trg) in conversions.items():
        _, Out, _ = model(X, c_org, c_trg)
        Out1 = Out.squeeze(0).squeeze(0).detach().numpy()
        converted_numpy.append(Out1)
        converted_tensor.append(Out)
    return S.squeeze(0).detach().numpy(), T.squeeze(0).detach().numpy(), tuple(converted_numpy)
    if Visualize:
        
        titles = ["Source", "Source-Source", "Target-Target", "Target", "Target-Target", "Target-Source"]
        fig = plt.figure(figsize = (10,10))
        fig.add_subplot(2,3,1)
        plt.imshow(s.T)
        plt.title(titles[0])
        fig.add_subplot(2,3,4)
        plt.imshow(t.T)
        plt.title(titles[3])

        index = [2,3,5,6]

        for i, spectrogram in enumerate(converted_numpy):
            fig.add_subplot(2,3,index[i])
            plt.imshow(spectrogram)
            plt.title(titles[index[i]-1])
        plt.show()
       

    if sound_out:
        
        if vocoder == "wavenet":
            wavenet = build_model().to("cpu")
            checkpoint = torch.load("Models/WaveNet/WaveNetVC_pretrained.pth", map_location=torch.device("cpu"))
            wavenet.load_state_dict(checkpoint["state_dict"])

            for i, spectrogram in enumerate(converted_tensor):
                waveform = wavegen(wavenet, c = spectrogram.squeeze(0).squeeze(0))
                librosa.output.write_wav(f"convert{i+1}"+'.wav', waveform, sr=16000)

        else:
            for i, spectrogram in enumerate(converted_numpy):
                Generate(spectrogram.T)
    

model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/AutoVC/autoVC_full_wavenet_original_step200k.pt", map_location=torch.device("cpu"))
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
t = data[25]

S, T, (SS, ST, TT, TS)  = Conversion(s, t, model, vocoder = "wavernn", Visualize= False, sound_out= False)

A = ["source", "target", "Source_Source", "Source_Target", "Target_Target", "Target_Source"]
B = [S, T, SS, ST, TT, TS]
for i, a in enumerate(A):
    Generate(B[i].T, "ConvertedWavs/" + a + "1", voc_model)

end = time.time()

print(f"\nTotal time: {end - start}")




