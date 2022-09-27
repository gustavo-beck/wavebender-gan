import argparse
import json
import os
import re
import sys
import  random
import math
import numpy as np
import torch
from os import listdir
from os.path import isfile, join
from scipy.io.wavfile import read
import torch.utils.data
from torch.utils.data import DataLoader
from mlp import MLPModel
from wavebender_ae_extension import ConvAutoencoder
from scipy.io.wavfile import write
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import parselmouth
import sounddevice as sd
from hifi.env import AttrDict
from hifi.models import Generator
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT
import pandas as pd

MAX_WAV_VALUE = 32768.0

def standardization(ts, mean, std):
    ts -= mean
    ts /= std
    return ts

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    wav = np.copy(data)
    return torch.FloatTensor(wav)

def get_tacotron_mel(audio):
        stft = TacotronSTFT(filter_length=1024,
                                    hop_length=256,
                                    win_length=1024,
                                    sampling_rate=22050,
                                    mel_fmin=0.0, mel_fmax=8000.0)
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

def modify_features(file_path, is_modifiy, selected_feature, scale):
    features = features_control(file_path, is_modifiy, selected_feature, scale)
    return features

def compute_features(wave, scale):
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    f0_contour = wave.f0_contour(method='rapt', f0_max=8000)[0] # The number of glottal pulses in a second
    intensity = wave.intensity()[0] # Power carried by sound waves per unit area.
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    
    ts_features = [f1, f2, 
                   f0_contour, intensity,
                   spectral_centroid, spectral_slope]
    
    return ts_features


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3

def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(nrow, ncol,
                                sharex=True,
                                sharey=False)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes

def plot_features(ts, ts_o, scale):

    f_list = ["f1", "f2",
              "f0_contour", "intensity",
              "spectral_centroid", "spectral_slope"
            ]

    for idx, (f, f_0) in enumerate(zip(ts, ts_o)):
        plt.plot(range(len(f)), f, label="Modified {} with scale {}".format(f_list[idx], str(scale)))
        plt.plot(range(len(f_0)), f_0, label="Original {}".format(f_list[idx]))
        plt.legend()
        plt.show()

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
    
def features_control(file_path, scale = 1.0):

    wave_original = Waveform(os.path.join(file_path), sample_rate=22050)
    ts_features_manipulation = compute_features(wave_original, scale)
    
    features_selection = ["f1", "f2",
                          "f0_contour", "intensity",
                          "spectral_centroid", "spectral_slope"
                         ]

    output_features = ["f1", "f2",
                          "f0_contour", "intensity",
                          "spectral_centroid", "spectral_slope",
                          "f0_mask"
                         ]

    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())
    features_indeces = [features.index(f) for f in features_selection]

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    features_padded = torch.FloatTensor(len(ts_features_manipulation) + 1, len(ts_features_manipulation[0])).zero_()

    for idx, feature in enumerate(ts_features_manipulation):
        feature = torch.tensor(feature)
        length = feature.size(0)
        feature_name = features_selection[idx]
        # Normalize/Standardize values
        if feature_name == "f0_contour":
            # Generate mask
            mask_zeros = torch.zeros(feature.shape)
            mask_ones = torch.ones(feature.shape)
            f0_mask = torch.where(feature > 0, mask_ones, mask_zeros)
            features_padded[-1]= f0_mask

            # Perform interpolation of log
            feature = torch.log(feature)
            feature[feature == -np.inf] = np.nan
            feature = feature.numpy()
            nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
            feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            features_padded[idx] = torch.FloatTensor(standardization(feature, mean, std))
        else:
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            features_padded[idx] = torch.FloatTensor(standardization(feature.float(), mean, std))

    return features_padded.T

def infer(params):

    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    # Load WaveBenderNet
    num_features = 7
    output_dim = 80
    mlp = MLPModel(num_features, output_dim, dropout=0.20, n_hid=128)
    mlp.load_state_dict(torch.load(params.mlp_path))
    mlp.to(device).eval()

    # Load AutoEncoder
    generator = ConvAutoencoder(params)
    generator.load_state_dict(torch.load(params.generator_path))
    generator.to(device)

    # Load WaveGlow
    hifi = Generator(h).to(device)
    state_dict_g = load_checkpoint(params.hifi_file, device)
    hifi.load_state_dict(state_dict_g['generator'])
    hifi.eval()
    hifi.remove_weight_norm()
    
    # Load the data features
    file_name = params.file_name # Example: 'LJ001-0002.wav'
    test_features = features_control("data/wavs/" + file_name, params.scale)

    # Infer and save Audio
    # Predict mel MLP
    print(test_features.shape)
    mel_mlp = mlp(test_features.to(device))
    print(mel_mlp.shape)

     # Store mel-spectrograms
    fname = os.path.join(params.infered_mel_spectrograms_path + file_name[:-4])
    plt.imshow(mel_mlp.T.cpu().detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
    plt.savefig(fname + ".png")
    plt.close()

    # Enhance mel Encoder
    mel = generator(mel_mlp.T.unsqueeze(0).unsqueeze(0))
    mel = mel[0, 0,:,:]
    
    # Predict audio from wavebender + gan
    with torch.no_grad():
        y_g_hat = hifi(mel.unsqueeze(0))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    audio_path = os.path.join(
        params.infered_audio_path, "{}_synthesis.wav".format(file_name))
    write(audio_path, params.sampling_rate, audio)

if __name__ == "__main__":
    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--file_name", 
                        help="Define the directory to read the the features of test files",
                        type=str,
                        default="LJ026-0014.wav")
    parser.add_argument("--mlp_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="mlp.pt")
    parser.add_argument("--generator_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="generator_checkpoints/generator_extension.pt")
    parser.add_argument("--waveglow_path", 
                        help="Define the waveglow model path",
                        type=str,
                        default="waveglow_256channels_universal_v5.pt")
    parser.add_argument("--n_channels_in", 
                        help="How many inputs we pass through",
                        type=int,
                        default= 11)
    parser.add_argument("--infered_mel_spectrograms_path",
                        help="Infered Mel-Spectrograms from Wavebender Net",
                        type=str,
                        default="wavebender_mel_spectograms/infered_test/mel_spectrograms/")
    parser.add_argument("--infered_audio_path",
                        help="Infered Audio from WaveGlow",
                        type=str,
                        default="wavebender_mel_spectograms/infered_test/audios/")
    parser.add_argument("--store_mel",
                        help="Infered Mel-Spectrograms from Wavebender Net",
                        type=bool,
                        default=True)
    parser.add_argument("--n_channels_out", 
                        help="How many channels to predict",
                        type=int,
                        default= 80)
    parser.add_argument("-sig", "--sigma", default=0.6, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')
    parser.add_argument("-c", "--control",
                        default=True,
                        type=bool,
                        help='Define if you want to modify the features')
    parser.add_argument("-s", "--scale",
                        default=1.2,
                        required = True,
                        type=float,
                        help='Define the scale to modify the feature')
    parser.add_argument('--config_file', default="hifi/config.json")
    parser.add_argument('--hifi_file', default="hifi/hifi_model")

    args = parser.parse_args()

    with open(args.config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    infer(args)


