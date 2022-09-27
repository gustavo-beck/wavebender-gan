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

class ParamObject(object):
    pass

def standardization(ts, mean, std):
    ts -= mean
    ts /= std
    return ts

def get_tacotron_mel(audio):
        stft = TacotronSTFT(filter_length=1024,
                                    hop_length=256,
                                    win_length=1024,
                                    sampling_rate=22050,
                                    mel_fmin=0.0, mel_fmax=8000.0)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

def compute_features(wave):
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    f0_contour = wave.f0_contour(method='rapt', f0_max=8000)[0] # The number of glottal pulses in a second
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    
    ts_features = [f1, f2,
                    f0_contour,
                    spectral_centroid, spectral_slope
                    ]

    return ts_features

def plot_features(ts, ts_o, scale, position, f_list):

    mat_mse_aux = []

    for idx, (f, f_0) in enumerate(zip(ts, ts_o)):
        plt.plot(range(len(f)), f, label="{} when {} is modified by a scale of {}".format(f_list[idx], f_list[int(position)], str(scale)))
        plt.plot(range(len(f_0)), f_0, label="Original {}".format(f_list[idx]))
        plt.legend()
        plt.savefig("wavebender_mel_spectograms/infered_test/audios/" + f_list[idx] + ".png")
        plt.close()

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
    
def features_control(file_path, features_selection, params, device, scale = 1.0, position = 0):

    wave_original = Waveform(os.path.join(file_path), sample_rate=22050)

    # Get augmented audio, mel and features
    wav = np.copy(wave_original.waveform)
    mel_tacotron = get_tacotron_mel(torch.FloatTensor(wav))

    # Time-series
    ts_features = compute_features(wave_original)
    ts_features_original = compute_features(wave_original)

    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    norm_ts = torch.FloatTensor(len(features_selection) + 1, len(ts_features[0])).zero_()

    for idx, feature in enumerate(ts_features):
        # Normalize/Standardize values
        if features_selection[idx] == "f0_contour":
            # Generate mask
            feature = torch.tensor(feature)
            mask_zeros = torch.zeros(feature.shape)
            mask_ones = torch.ones(feature.shape)
            f0_mask = torch.where(feature > 0, mask_ones, mask_zeros)
            norm_ts[-1] = f0_mask

            # Perform interpolation of log
            if features_selection[int(position)] == "f0_contour":
                feature *= scale
            feature = torch.log(feature)
            feature[feature == -np.inf] = np.nan
            feature = feature.numpy()
            nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
            feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            norm_ts[idx] = torch.FloatTensor(standardization(feature, mean, std))
        else:
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            if int(position) == idx:
                feature[-80:] *= scale
                norm_ts[idx] = torch.FloatTensor(standardization(feature, mean, std))
            else:
                norm_ts[idx] = torch.FloatTensor(standardization(feature, mean, std))
    
    if features_selection[int(position)] == "f1" and scale != 1.0:

        params_dict = ParamObject()
        params_dict.n_channels_in = 1
        params_dict.n_channels_out = 1

        # Load WaveBenderNet
        print("LOAD F1 to F2 MODEL")
        f1tof2 = WaveBenderNet(ResidualBlock, params_dict)
        f1tof2.load_state_dict(torch.load(params.f1tof2_path))
        f1tof2.to(device).eval()
        norm_ts[int(position + 1)] = f1tof2(norm_ts[int(position)].unsqueeze(0).unsqueeze(0).to(device))

    if features_selection[int(position)] == "f2" and scale != 1.0:
        params_dict = ParamObject()
        params_dict.n_channels_in = 1
        params_dict.n_channels_out = 1

        # Load WaveBenderNet
        print("LOAD F2 to F1 MODEL")
        # Load WaveBenderNet
        f2tof1 = WaveBenderNet(ResidualBlock, params_dict)
        f2tof1.load_state_dict(torch.load(params.f2tof1_path))
        f2tof1.to(device).eval()
        norm_ts[int(position - 1)] = f2tof1(norm_ts[int(position)].unsqueeze(0).unsqueeze(0).to(device))

    return norm_ts, mel_tacotron, ts_features_original

def infer(params):

    features_selection = ["f1", "f2",
                    "f0_contour",
                    "spectral_centroid", "spectral_slope"
                    ]

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
    test_features, mel_tacotron, ts_features_original = features_control("mos_test/" + file_name, features_selection, params, device, params.scale, params.position)

    # Infer and save Audio
    # Predict mel Wavebender
    mel_wavebender = wavebender(test_features.unsqueeze(0).to(device)).unsqueeze(1)

    # Enhance mel Encoder
    mel = generator(mel_wavebender)
    mel = mel[0, 0,:,:]

    if params.store_mel:
        # Store mel-spectrograms
        fname = os.path.join(params.infered_mel_spectrograms_path + file_name[:-4])
        plt.figure(figsize = (20,10))
        plt.imshow(mel.cpu().detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + "_wavebender_gan.png")
        plt.close()
        plt.figure(figsize = (20,10))
        plt.imshow(mel_wavebender.cpu().detach().numpy()[0][0],interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + "_wavebender.png")
        plt.close()
        plt.figure(figsize = (20,10))
        plt.imshow(mel_tacotron.detach().numpy(),interpolation='none',cmap=plt.cm.jet,origin='lower')
        plt.savefig(fname + "_tacotron.png")
        plt.close()
    
    # Predict audio from wavebender + gan
    with torch.no_grad():
        y_g_hat = hifi(mel.unsqueeze(0))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    audio_path = os.path.join(
        params.infered_audio_path, "{}_wavebender_gan_synthesis.wav".format(file_name))
    write(audio_path, params.sampling_rate, audio)
    print(audio_path)

    # Plot features
    ts_features = compute_features(Waveform(os.path.join(
        params.infered_audio_path, "{}_wavebender_gan_synthesis.wav".format(file_name)), sample_rate=22050))

    # plot_features(ts_features, ts_features_original, params.scale, params.position, features_selection)

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
                        default="wavebender_gan_thesis.pt")
    parser.add_argument("--f1tof2_path", 
                        help="Define the f1tof2 model path",
                        type=str,
                        default="f1tof2.pt")
    parser.add_argument("--f2tof1_path", 
                        help="Define the f2tof1 model path",
                        type=str,
                        default="f2tof1.pt")
    parser.add_argument("--generator_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="generator_checkpoints/generator_final_thesis.pt")
    parser.add_argument("--waveglow_path", 
                        help="Define the waveglow model path",
                        type=str,
                        default="waveglow_256channels_universal_v5.pt")
    parser.add_argument("--n_channels_in", 
                        help="How many inputs we pass through",
                        type=int,
                        default= 6)
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
    parser.add_argument("-p", "--position",
                        default=0,
                        required = True,
                        type=float,
                        help='Define the position to modify the specific feature')
    parser.add_argument('--config_file', default="hifi/config.json")
    parser.add_argument('--hifi_file', default="hifi/hifi_model")

    args = parser.parse_args()

    with open(args.config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    infer(args)


