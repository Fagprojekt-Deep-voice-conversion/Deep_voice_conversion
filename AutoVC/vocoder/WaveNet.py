import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import torch
from tqdm import tqdm
from hparams import hparams_autoVC as hparams
from wavenet_vocoder import builder

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def build_model():
    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=hparams.legacy,
    )
    return model


def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    """

    model.eval()
    model.make_generation_fast_()

    Tc = c.shape[0]
    upsample_factor = hparams.hop_size
    # Overwrite length according to feature size
    length = Tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


