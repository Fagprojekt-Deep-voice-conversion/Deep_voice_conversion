import argparse
import glob
import torch
import librosa
import numpy as np
import os
from os.path import join, basename
from shutil import copy
from data_loader import to_categorical
from model import Generator
from utils import world_decompose, pitch_conversion, world_encode_spectral_envelop, world_speech_synthesis, wav_padding


class ConvertDataset(object):
    """Dataset for conversion."""
    def __init__(self, config, src_spk, trg_spk):
        speakers = config.speakers
        spk2idx = dict(zip(speakers, range(len(speakers))))
        assert trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {trg_spk}.'

        self.src_spk = src_spk
        self.trg_spk = trg_spk

        # Source speaker locations.
        self.src_mc_files = sorted(glob.glob(join(config.test_data_dir, f'{self.src_spk}*.npy')))
        self.src_spk_stats = np.load(join(config.train_data_dir, f'{self.src_spk}_stats.npz'))
        self.src_wav_dir = f'{config.wav_dir}/{self.src_spk}'

        # Target speaker locations.
        self.trg_spk_stats = np.load(join(config.train_data_dir, f'{self.trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        self.spk_idx = spk2idx[self.trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=4):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.src_mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))

            batch_data.append(wavfile_path)

        return batch_data


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple=4)


def convert(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period = 16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    generator = Generator().to(device)
    g_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))

    # for all possible speaker pairs in config.speakers
    for i in range(0, len(config.speakers)):
        for j in range(0, len(config.speakers)):
            if i != j:
                target_dir = join(config.convert_dir,
                                  str(config.resume_iters),
                                  f'{config.speakers[i]}_to_{config.speakers[j]}')

                os.makedirs(target_dir, exist_ok=True)

                # Load speakers
                data_loader = ConvertDataset(config, src_spk=config.speakers[i], trg_spk=config.speakers[j])
                print('---------------------------------------')
                print('Source: ', config.speakers[i], ' Target: ', config.speakers[j])
                print('---------------------------------------')

                # Read a batch of testdata
                src_test_wavfiles = data_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
                src_test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in src_test_wavfiles]

                with torch.no_grad():
                    for idx, wav in enumerate(src_test_wavs):
                        print(f'({idx}), file length: {len(wav)}')
                        wav_name = basename(src_test_wavfiles[idx])

                        # convert wav to mceps
                        f0, _, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0,
                                                        mean_log_src=data_loader.logf0s_mean_src,
                                                        std_log_src=data_loader.logf0s_std_src,
                                                        mean_log_target=data_loader.logf0s_mean_trg,
                                                        std_log_target=data_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        print("Before being fed into G: ", coded_sp.shape)
                        coded_sp_norm = (coded_sp - data_loader.mcep_mean_src) / data_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
                        spk_conds = torch.FloatTensor(data_loader.spk_c_trg).to(device)

                        # generate converted speech
                        coded_sp_converted_norm = generator(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * data_loader.mcep_std_trg + data_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        print("After being fed into G: ", coded_sp_converted.shape)

                        # convert back to wav
                        wav_transformed = world_speech_synthesis(f0=f0_converted,
                                                                 coded_sp=coded_sp_converted,
                                                                 ap=ap,
                                                                 fs=sampling_rate,
                                                                 frame_period=frame_period)
                        wav_id = wav_name.split('.')[0]

                        # SAVE TARGET SYNTHESIZED
                        librosa.output.write_wav(join(target_dir, f'{wav_id}-vcto-{data_loader.trg_spk}.wav'),
                                                 wav_transformed,
                                                 sampling_rate)

                        # SAVE COPY OF TARGET REFERENCE
                        wav_num = wav_name.split('.')[0].split('_')[1]
                        copy(f'{config.wav_dir}/{config.speakers[j]}/{config.speakers[j]}_{wav_num}.wav', target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=10, help='Dimension of speaker labels.')
    parser.add_argument('--num_converted_wavs', type=int, default=8, help='Number of wavs to convert.')
    parser.add_argument('--resume_iters', type=int, default=None, help='Step to resume for testing.')
    parser.add_argument('--speakers', type=str, nargs='+', required=True, help='Speakers to be converted.')

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')

    config = parser.parse_args()
    print(config)

    if config.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    if len(config.speakers) < 2:
        raise RuntimeError("Need at least 2 speakers to convert audio.")

    convert(config)
