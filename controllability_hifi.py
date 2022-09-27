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
from wavebendernet import WaveBenderNet, ResidualBlock
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

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    wav = np.copy(data)
    return torch.FloatTensor(wav)

def modify_features(file_path, is_modifiy, selected_feature, scale):
    features = features_control(file_path, is_modifiy, selected_feature, scale)
    return features

def compute_features(wave):
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    log_energy = wave.log_energy_slidingwindow()[0] # Easier to interpret than the signal 
    zcr = wave.zerocrossing_slidingwindow()[0] # Can be interpreted as a measure of the noisiness of a signal
    f0_contour = wave.f0_contour(method='rapt', f0_max=8000)[0] # The number of glottal pulses in a second
    intensity = wave.intensity()[0] # Power carried by sound waves per unit area.
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    spectral_spread = wave.spectral_spread()[0] # Associated with the bandwidth of the signal. E.g. Individual tonal sounds with isolated peaks will result in a low spectral spread.

    ts_features = [f1, f2, f3, f4,
                    log_energy, zcr,
                    f0_contour, intensity,
                    spectral_centroid, spectral_slope,
                    spectral_spread
                    ]
    
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

    f_list = ["f1", "f2", "f3", "f4",
              "log_energy", "zcr",
              "f0_contour", "intensity",
              "spectral_centroid", "spectral_slope",
              "spectral_spread"
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
    sound = parselmouth.Sound(os.path.join(file_path))
    factor = scale
    manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
    sound_octave_up = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    sound_octave_up.save("manipulation.wav", "WAV")
    wave = Waveform("manipulation.wav", sample_rate=22050)
    wave_original = Waveform(os.path.join(file_path), sample_rate=22050)
    
    # Test waveglow
    audio = load_wav_to_torch("manipulation.wav")
    mel_tacotron  = get_tacotron_mel(audio)

    # Time-series
    ts_features = compute_features(wave)
    ts_features_original = compute_features(wave_original)
    
    # Plot features
    # plot_features(ts_features, ts_features_original, scale)

    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    norm_ts = torch.FloatTensor(len(ts_features), len(ts_features[0])).zero_()

    for idx, ts in enumerate(ts_features):
        mean = global_stats[features[idx]]["mean"]
        std = global_stats[features[idx]]["std"]
        norm_ts[idx] = torch.FloatTensor(standardization(ts, mean, std))

    return torch.FloatTensor(norm_ts), mel_tacotron

def infer(params):

    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    # Load WaveBenderNet
    wavebender = WaveBenderNet(ResidualBlock, params)
    wavebender.load_state_dict(torch.load(params.wavebender_path))
    wavebender.to(device).eval()

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
    test_features, mel_tacotron = features_control("data/wavs/" + file_name, params.scale)

    # Infer and save Audio
    # Predict mel Wavebender
    mel_wavebender = wavebender(test_features.unsqueeze(0).to(device)).unsqueeze(1)

    # Enhance mel Encoder
    mel = generator(mel_wavebender)
    mel = mel[0, 0,:,:]

    if params.store_mel:
        # Store mel-spectrograms
        fname = os.path.join(params.infered_mel_spectrograms_path + file_name[:-4])
        plt.imshow(mel.cpu().detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + ".png")
        plt.close()
        plt.imshow(mel_tacotron.detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + "_tacotron.png")
        plt.close()
        plt.imshow(mel_wavebender[0, 0,:,:].cpu().detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + "_wavebender.png")
        plt.close()

    # Predict audio from wavebender + gan
    with torch.no_grad():
        y_g_hat = hifi(mel_wavebender[0, 0,:,:].unsqueeze(0))
        audio_wave = y_g_hat.squeeze()
        audio_wave = audio_wave * MAX_WAV_VALUE
        audio_wave = audio_wave.cpu().numpy().astype('int16')
    audio_wave_path = os.path.join(
        params.infered_audio_path, "{}_wavebender_synthesis.wav".format(file_name))
    write(audio_wave_path, params.sampling_rate, audio_wave)
    print(audio_wave_path)
    
    # Predict audio from wavebender + gan
    with torch.no_grad():
        y_g_hat = hifi(mel.unsqueeze(0))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    audio_path = os.path.join(
        params.infered_audio_path, "{}_synthesis.wav".format(file_name))
    write(audio_path, params.sampling_rate, audio)
    print(audio_path)

    # Predict audio from tacotron2
    with torch.no_grad():
        y_g_hat = hifi(mel_tacotron.unsqueeze(0).to(device))
        audio_waveglow = y_g_hat.squeeze()
        audio_waveglow = audio_waveglow * MAX_WAV_VALUE
        audio_waveglow = audio_waveglow.cpu().numpy().astype('int16')
    audio_waveglow_path = os.path.join(
        params.infered_audio_path, "{}_tacotron_synthesis.wav".format(file_name))
    write(audio_waveglow_path, params.sampling_rate, audio_waveglow)
    print(audio_waveglow_path)

if __name__ == "__main__":
    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--file_name", 
                        help="Define the directory to read the the features of test files",
                        type=str,
                        default="LJ026-0014.wav")
    parser.add_argument("--wavebender_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="wavebender_augmented.pt")
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


