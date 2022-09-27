import os
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
from os import listdir
from os.path import isfile, join

audio_path = "wavebender_mel_spectograms/infered_test/audios/"
audio_type = "LJ001-0002"

CB91_Blue = '#2CBDFE'
CB91_Green = 'springgreen'
CB91_Red = '#DA6F6F'

# Normalize time series
def standardization(ts, mean, std):
    ts -= mean
    ts /= std
    return ts

# Extract time-series and continuos features
def ext_features(file_path):
    # Get .wav file
    wave = Waveform(path=os.path.join(file_path), sample_rate=22050)
    
    # Time-series
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    log_energy = wave.log_energy_slidingwindow()[0] # Easier to interpret than the signal 
    zcr = wave.zerocrossing_slidingwindow()[0] # Can be interpreted as a measure of the noisiness of a signal
    f0_contour = wave.f0_contour(method='rapt')[0] # The number of glottal pulses in a second
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
    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    norm_ts = []
    for idx, feature in enumerate(features):
        mean = global_stats[features[idx]]["mean"]
        std = global_stats[features[idx]]["std"]
        norm_ts.append(standardization(ts_features[idx], mean, std))

    return norm_ts

path = audio_path + audio_type

files = [f for f in listdir(path) if isfile(join(path, f))]

audio_features = {}
# Get global data stats (mean, std, max, min)
with open(os.path.join("global_stats.json")) as f:
    global_stats = json.load(f)
features = list(global_stats.keys())

for file in files:
    audio_features[file] = ext_features(path + "/" + file)

original_audio = audio_features[audio_type + ".wav"]
audio_features.pop(audio_type + ".wav", None)
files.pop(files.index(audio_type + ".wav"))
for file in files:
    print(file)
    for idx, feature in enumerate(audio_features[file]):
        print(features[idx])
        try:
            print(mean_squared_error(original_audio[idx], feature))
        except:
            print(mean_squared_error(original_audio[idx], feature[:-1]))

lenght = len(original_audio[0])
range_feature = range(lenght)

for idx, feature in enumerate(features):
    plt.figure(figsize=(15,10))
    mse_wavebender = mean_squared_error(original_audio[idx], audio_features["LJ001-0002_v4_synthesis.wav"][idx][:lenght])
    mse_glow = mean_squared_error(original_audio[idx], audio_features["LJ001-0002_v5_glow.wav_synthesis.wav"][idx][:lenght])
    plt.plot(range_feature, original_audio[idx], label="Original", c=CB91_Blue)
    plt.plot(range_feature, audio_features["LJ001-0002_v4_synthesis.wav"][idx][:lenght], label="WaveBender with MSE {0:.2f}".format(mse_wavebender), c=CB91_Green)
    plt.plot(range_feature, audio_features["LJ001-0002_v5_glow.wav_synthesis.wav"][idx][:lenght], label="Wavebender Glow with MSE {0:.2f}".format(mse_glow), c=CB91_Red)
    plt.title(feature)
    plt.legend()
    plt.savefig("consistency_plots/" + feature + ".png")
    plt.close()


