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
    return S, T


data, labels = DataLoad2("Test_Data")
data,labels  = data[:20], labels[:20]


s = data[0]
t = data[10]

model = Generator(32, 256, 512, 32).eval().to("cpu")
g_checkpoint = torch.load("Models/AutoVC/autovc_200k_average_wavenet.pt", map_location=torch.device("cpu"))
model.load_state_dict(g_checkpoint['model_state'])



ST, S  = Conversion(s, t, model, vocoder = "wavenet", Visualize= False, sound_out= True)

# Auto VC vocoder 

"""
model = build_model().to("cpu")
checkpoint = torch.load("Models/WaveNet/WaveNetVC_pretrained.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["state_dict"])


waveform = wavegen(model, c = S[0].squeeze(0))
librosa.output.write_wav("source"+'.wav', waveform, sr=16000)
waveform = wavegen(model, c = T[0].squeeze(0))
librosa.output.write_wav("target"+'.wav', waveform, sr=16000)

waveform = wavegen(model, c = ST.squeeze(0).squeeze(0))
librosa.output.write_wav("conversion "+'.wav', waveform, sr=16000)



"""


