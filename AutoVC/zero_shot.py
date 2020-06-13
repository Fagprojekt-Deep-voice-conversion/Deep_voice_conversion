from conversion import *
import torch
from Preprocessing_WAV import WaveRNN_Mel


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
    Zero_shot("2.wav", "../Experiment/Survey-app/voice-conversion-survey/www/persons/mette/mette_183.wav", model, voc_model, ".")
