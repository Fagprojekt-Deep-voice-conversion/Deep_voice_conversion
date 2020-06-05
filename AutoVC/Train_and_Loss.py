"""
The loss and training script for AutoVC (home made but inspired by AutoVC paper)

"""
import os, sys
import numpy as np
import torch
from tqdm import tqdm
from Speaker_identity import SpeakerIdentity
from Preprocessing_WAV import Mel_Batch
import pickle
from dataload import DataLoad2

path = sys.path[0]
os.chdir(path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def TrainLoader(Data, labels, batch_size = 2, shuffle = True, num_workers = 1, pin_memory = False, vocoder = "autovc"):

    Data, labels = np.array(Data)[np.argsort(labels)], np.array(labels)[np.argsort(labels)]
    embeddings, uncorrupted = SpeakerIdentity(Data)
    labels = labels[uncorrupted]
    emb = []
    # for person in sorted(set(labels)):
        # index = np.where(labels == person)
    #     X = embeddings[index].cpu()
    #     X = X.mean(0).unsqueeze(0).expand(len(index[0]), -1)
    #     emb.append(X)
    # emb = torch.cat(emb, dim = 0).to(device)
    emb = embeddings
    Mels, uncorrupted = Mel_Batch(list(Data[uncorrupted]), vocoder = vocoder)
    emb = emb[uncorrupted]


    C = torch.utils.data.DataLoader(ConcatDataset(Mels, emb), shuffle = shuffle,
                                    batch_size = batch_size, collate_fn = collate,
                                    num_workers = num_workers,
                                    pin_memory = pin_memory)
    return C, uncorrupted

def collate(batch):
    batch = list(zip(*batch))
    lengths = torch.tensor([t.shape[1] for t in batch[0]])
    m = lengths.max()
    Mels = []
    for t in batch[0]:
        pad = torch.nn.ConstantPad2d((0, 0, 0, m - t.size(1)), 0)
        t = pad(t)
        Mels.append(t)
    Mels = torch.cat(Mels, dim = 0)
    embeddings = torch.cat([t.unsqueeze(0) for t in batch[1]], dim = 0)

    return [Mels, embeddings]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



def loss(output, target, model, mu = 1, lambd = 1):
    """
    Loss function as proposed in AutoVC
    L = Reconstruction Error + mu * Prenet reconstruction Error + lambda * content Reconstruction error
    mu and lambda are set to 1 in the paper.

    params:
    outputs: as for now a simple try... A list of outputs of the batch:
        - batch_size * decoder_outputs
        - bacth_size * postnet_outputs
        - batch_size * content codes
    target: A list of the targets
        - batch_size * orignal Mel spectrograms
        - batch_size * original Speaker identiy embedding
    mu, lambda: model parameters mu and lambda

    returns the loss function as proposed in AutoVC
    """

    """ Zips output ... """
    out_decoder, out_post, codes = output[0].squeeze(1), output[1].squeeze(1), output[2]
    X, c_org = target[0], target[1]
    ReconCodes = model(out_post, c_org, None)
    """ 
    Reconstruction error: 
        The mean of the squared p2 norm of (Postnet outputs - Original Mel Spectrograms)
    """
    err_reconstruct  = (torch.norm(out_post - X, p = 2, dim = (1,2))**2).mean()
    """
    Prenet Reconstruction error
        The mean of the squared p2 norm of (Decoder outputs - Original Mel Spectrograms)
    """
    err_reconstruct0 = (torch.norm(out_decoder - X, p = 2, dim = (1,2))**2).mean()

    """
    Content reconstruction Error
        The mean of the p1 norm of (Content codes of postnet output - Content codes)
    """
    err_content      = torch.norm(ReconCodes - codes, p = 1, dim = 0).mean()

    return err_reconstruct + mu * err_reconstruct0 + lambd * err_content


# Check for GPU
""" A bit of init stuff ... Checking for GPU and loading data """



def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
    # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(
        step * warmup_steps ** -1.5, step ** -0.5)
    return lr

def load_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()].view(param.size()))
        offset += param.nelement()

def flatten_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()], 0)


def Train(model, trainloader, init_lr, n_steps, save_every, models_dir, loss_dir, model_path_name, loss_path_name):
    if torch.cuda.is_available():
        print(f"Training beginning on {torch.cuda.get_device_name(0)}")
    else:
        print(f"Training beginning on cpu")

    step = 1
    ema = 0.9999
    running_loss = []

    loss_fpath = loss_dir + "/" + loss_path_name
    optimiser = torch.optim.Adam(model.parameters(), lr=init_lr, betas = (0.9, 0.999),
                                 eps = 1e-8, weight_decay=0.0, amsgrad = False)
    model.train()
    avg_params = flatten_params(model)
    while step < n_steps:
        for X, c_org in tqdm(trainloader):

            """ Outputs and loss"""
            mel, post, codes = model(X, c_org, c_org)
            error = loss([mel, post, codes], [X, c_org], model)

            """ Zeros the gradient for every step """
            """ Computes gradient and do optimiser step"""
            optimiser.zero_grad()
            error.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1,1)
            optimiser.step()
            optimiser.param_groups[0]["lr"] = noam_learning_rate_decay(init_lr, step)
            avg_params = ema * avg_params + (1-ema) * flatten_params(model)
            step += 1


            if step % 10 == 0:
                """ Append current error to L for plotting """
                r = error.cpu().detach().numpy()
                running_loss.append(r)
                pickle.dump(running_loss, open(loss_fpath, "wb"))

            if step % save_every == 0:
                original_param = flatten_params(model)
                load_params(model, avg_params)
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimiser.state_dict(),
                }, models_dir + "/" + model_path_name + "average_"+ f"_step{step / 1000}k" ".pt")
                load_params(model, original_param)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimiser.state_dict(),
                }, models_dir + "/" + model_path_name + "_original" +f"_step{step / 1000}k" ".pt")

            if step >= n_steps:
                break



    pickle.dump(running_loss, open(loss_fpath, "wb"))
    print("Saving the model (step %d)" % step)
    torch.save({
        "step": step + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimiser.state_dict(),
    }, models_dir + "/" + model_path_name + "_original" + f"_step{step / 1000}k" ".pt")
    load_params(model, avg_params)

    torch.save({
        "step": step + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimiser.state_dict(),
    }, models_dir + "/" + model_path_name + "average_" + f"_step{step / 1000}k" ".pt")








