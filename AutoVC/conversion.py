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
import pandas as pd

def Instantiate_Models(model_path,  vocoder = "wavernn"):
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
                            mode='MOL').to(device)

        voc_model.load('Models/WaveRNN/WaveRNN_Pretrained.pyt')
    else:
        raise RuntimeError("No vocoder chosen")

    # Prepare AutoVC model
    model = Generator(32, 256, 512, 32).eval().to(device)
    g_checkpoint = torch.load(model_path, map_location=torch.device(device))
    if vocoder == "wavenet":
        model.load_state_dict(g_checkpoint['model'])
    else:
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




def Conversion(source, target, model, voc_model, T_emb = None, voc_type = "wavernn", task = None, subtask = None, exp_folder = None):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if voc_type == "wavernn":
        s = WaveRNN_Mel(source)
    else:
        s = AutoVC_Mel(source)

    S = torch.from_numpy(s.T).unsqueeze(0).to(device)
    
    S_emb, T_emb = embed(source).to(device), embed(target).to(device) if T_emb is None else T_emb.to(device)
    
    conversions = {"source": (S, S_emb, S_emb), "Converted": (S, S_emb, T_emb)}
    

    for key, (X, c_org, c_trg) in conversions.items():
        if key == "Converted":
            _, Out, _ = model(X, c_org, c_trg)
            name = source.split("/")[-1].split(".")[0] + "_to_" + (target.split("/")[-1].split(".")[0] if "/" in target else target)
            path = f"{exp_folder}/AutoVC/{task}/{subtask}/{name}"
        else:
            Out = X.unsqueeze(0)
            name = eval(key).split("/")[-1].split(".")[0]
            name1 = name.split("_")[0]
            path = f"{exp_folder}/AutoVC/persons/{name1}/{name}"
        
        
        
        print(f"\n Generating {key} sound")
        Generate(Out, path, voc_model, voc_type)


def Experiment(Model_path, train_length = None, test_data = None, name_list = None, test_size = 24, experiment = None):
# Load data about gender and language and store in dictionary
    dictionary = {}

    X = pd.read_csv(name_list, header = None)
    for i, x in enumerate(X.iloc[:,0]):
        dictionary.update({x: [X.iloc[i, 1], X.iloc[i, 2]]})
	
    
    (_, _), (data, labels) = DataLoad2(test_data, test_size= test_size)
    data, labels = np.array(data), np.array(labels)

    X = pd.read_csv("../data/good_voices.csv", header  = None)
    voices = {}
    for i, x in enumerate(X.iloc[:,0]):
        voices.update({x: X.iloc[i, 1:].values})
    for key, value in voices.items():
        index = np.where([labels == key])[1]

        index = index[np.invert([np.any([f"_{x}."  in wav for x in list(map(str, voices[key]))]) for wav in data[index]])]
        data = np.delete(data, index)
        labels = np.delete(labels, index)
	
    model, voc_model = Instantiate_Models(model_path = Model_path, vocoder = "wavernn")

    persons = np.array([person for person, _ in dictionary.items()])


    
    for source in persons:
        targets = persons[persons != source]

        for s in data[labels == source]:
          

            for target in targets:
                if (dictionary[source][1] == dictionary[target][1]) and train_length is None:
                    task = dictionary[source][1] + "_" + dictionary[target][1]
                    subtask = dictionary[source][0] + "_" + dictionary[target][0]

                    T_emb = torch.cat([embed(t) for t in data[labels == target]]).mean(0).unsqueeze(0);print(source, target)

                    Conversion(s, target, model, voc_model, T_emb = T_emb, task = task, subtask = subtask, voc_type="wavernn", exp_folder = experiment)
                
                elif train_length is not None:
                    task = train_length

                    if (dictionary[source][1] == "Danish" and dictionary[target][1] == "Danish") and (dictionary[source][0] == "Male" and dictionary[target][0] == "Male"):
                        subtask = "Male_Male"
                        T_emb = torch.cat([embed(t) for t in data[labels == target]]).mean(0).unsqueeze(0);print(source, target)
                        Conversion(s, target, model, voc_model, T_emb = T_emb, task = task, subtask = subtask, voc_type="wavernn", exp_folder = experiment)
                
                    
                


            





if __name__ == "__main__":
    # (data, labels), (_, _) = DataLoad2("../data/Test_Data")
    # model, voc_model = Instantiate_Models(model_path = 'autoVC_seed40_200k.pt', vocoder = "wavernn")
    # source, target = data[0], data[39]
    # Conversion(source, target, model, voc_model, voc_type = "wavernn", task = "English_English", subtask = "Male_Male")


    Experiment(Model_path = "Models/AutoVC/AutoVC_seed40_200k.pt", train_length =  None, test_data = "../data/test_data", name_list = "../data/persons2.csv", test_size = 11, experiment = "../Experiment" )
   
    

    
    
    
    

