import argparse
import json
import os
import re
import sys
import  random
import math
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from tqdm import tqdm
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import parselmouth
import sounddevice as sd
from sklearn.metrics import mean_squared_error

def normalization(ts, minimum, maximum):
    norm = ts - minimum
    ts /= (maximum - minimum)
    return ts

def compute_features(wave, features_selection, global_stats):
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    f0_contour = wave.f0_contour(method='rapt', f0_max=8000)[0] # The number of glottal pulses in a second
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    
    ts_features = [f1, f2,
                    f0_contour,
                    spectral_centroid, spectral_slope
                    ]
    
    norm_ts = []
    for idx, feature in enumerate(ts_features):
        if features_selection[idx] == "f0_contour":
            feature = np.log(feature)
            feature[feature == -np.inf] = np.nan
            nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
            feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
            norm_ts.append(normalization(feature, global_stats[features_selection[idx]]['min'], global_stats[features_selection[idx]]['max']))
        else:
            norm_ts.append(normalization(feature, global_stats[features_selection[idx]]['min'], global_stats[features_selection[idx]]['max']))

    return norm_ts

def plot_features(ts, ts_o, f_list):

    mat_mse_aux = []

    for idx, (f, f_0) in enumerate(zip(ts, ts_o)):
        plt.plot(range(len(f)), f, label="Synthesized {} ".format(f_list[idx]))
        plt.plot(range(len(f_0)), f_0, label="Original {}".format(f_list[idx]))
        plt.legend()
        plt.show()
    
def read_audios(features_selection, file_path):

    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)

    # Get files names
    waveforms = {}
    for file in os.listdir(file_path):
        if file.endswith(".wav"):
            # Read audio
            wave_original = Waveform(os.path.join(file_path, file), sample_rate=22050)
            if file[5:].split('.')[0] in waveforms:
                waveforms[file[5:].split('.')[0]].append(compute_features(wave_original, features_selection, global_stats))
            else:
                waveforms[file[5:].split('.')[0]] = [compute_features(wave_original, features_selection, global_stats)]
    
    audio_files = list(waveforms.keys())

    mse_hifi = {}
    mse_wavebender = {}

    for idx, x in enumerate(audio_files):
        original_audio = waveforms[x][0]
        hifi = waveforms[x][1]
        wavebender = waveforms[x][2]
        '''if idx == 9: 
            plot_features(wavebender, original_audio, features_selection)'''
        for y in range(len(original_audio)):
            if y < 3:
                if features_selection[y] in mse_hifi:
                    mse_hifi[features_selection[y]].append(mean_squared_error(original_audio[y], hifi[y]))
                    mse_wavebender[features_selection[y]].append(mean_squared_error(original_audio[y], wavebender[y]))
                else:
                    mse_hifi[features_selection[y]] = [mean_squared_error(original_audio[y], hifi[y])]
                    mse_wavebender[features_selection[y]] = [mean_squared_error(original_audio[y], wavebender[y])]
            else:
                if features_selection[y] in mse_hifi:
                    mse_hifi[features_selection[y]].append(mean_squared_error(original_audio[y], hifi[y][:-1]))
                    mse_wavebender[features_selection[y]].append(mean_squared_error(original_audio[y], wavebender[y][:-1]))
                else:
                    mse_hifi[features_selection[y]] = [mean_squared_error(original_audio[y], hifi[y][:-1])]
                    mse_wavebender[features_selection[y]] = [mean_squared_error(original_audio[y], wavebender[y][:-1])]
        
    box = plt.boxplot(mse_wavebender.values(), labels = features_selection, vert=False, patch_artist=True)
    for key in list(mse_wavebender.keys()):
        print(key)
        print(mse_wavebender[key])
        print(np.mean(mse_wavebender[key]))

    '''with open('mse_hifigan.txt', 'w') as f:
        f.write(json.dumps(str(mse_hifi)))'''
    
    with open('mse_wavebender_gan.txt', 'w') as f:
        f.write(json.dumps(str(mse_wavebender)))

    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("Wavebender GAN reconstruction errors")
    plt.ylabel("Low-level signal properties")
    plt.xlabel("Relative MSE")
    plt.tight_layout()
    plt.xlim([0, 0.045])
    plt.show()

    box = plt.boxplot(mse_hifi.values(), labels = features_selection, vert=False, patch_artist=True)

    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("HiFi-GAN reconstruction errors")
    plt.ylabel("Low-level signal properties")
    plt.xlabel("Relative MSE")
    plt.tight_layout()
    plt.xlim([0, 0.045])
    plt.show()

    return waveforms

def analyze(args):

    features_selection = ["f1", "f2",
                    "f0_contour",
                    "spectral_centroid", "spectral_slope"
                    ]

    read_audios(features_selection, args.file_name)

    

if __name__ == "__main__":

    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--file_name", 
                        help="Define the directory to read the audios",
                        type=str,
                        default="mos_test/")

    args = parser.parse_args()

    analyze(args)


