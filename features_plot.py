import argparse
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import os
import sys
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from os import walk
from sklearn.metrics import mean_squared_error
import torch 


CB91_Blue = '#2CBDFE'
CB91_Red = '#DA6F6F'

# Read .txt files
def read_files(input_path):
    with open(input_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files

def standardization(ts, mean, std):
    # Normal Z-score
    z_score = ts - mean
    z_score /= std

    return z_score

# Extract time-series and continuos features
def ext_features(file_path):
    # Get .wav file
    wave = Waveform(path=os.path.join(file_path), sample_rate=22050)
    
    # Time-series
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2 = formants[0], formants[1]
    f0_contour = wave.f0_contour(method='rapt', f0_max=1000)[0] # The number of glottal pulses in a second
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    
    ts_features = [f1, f2,
                    f0_contour,
                    spectral_centroid, spectral_slope,
                    ]

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    time_series = [ts.tolist() for ts in ts_features]

    return time_series

if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                        help='Input directory', default="data/wavs/")

    args = parser.parse_args()

    time_series_features = [
        "f1", "f2", "f0_contour",
        "spectral_centroid",
        "spectral_slope"
    ]

    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)

    # Extract features
    features = ext_features(args.input_dir + "LJ026-0014.wav")
    features_map = {}
    for idx, f in enumerate(features):
        features_map[time_series_features[idx]] = np.array(f)


    for ts in (time_series_features):
        mean = global_stats[ts]["mean"]
        std = global_stats[ts]["std"]
        if ts == "f0_contour":
            mask_zeros = np.zeros(len(features_map[ts]))
            mask_ones = np.ones(len(features_map[ts]))
            f0_mask = np.where(features_map[ts] > 0, mask_ones, mask_zeros)
            feature_aux = torch.log(torch.tensor(features_map[ts]))
            feature_aux[feature_aux == -np.inf] = np.nan
            feature_aux = feature_aux.numpy()
            nans, x = np.isnan(feature_aux), lambda z: z.nonzero()[0]
            feature_aux[nans] = np.interp(x(nans), x(~nans), feature_aux[~nans])
            plt.plot(standardization(feature_aux, mean, std), label="%s" % ("log-F0 contour (interpolated)"), c= CB91_Blue)
            plt.plot(f0_mask, label="%s" % ("F0 mask"), c= CB91_Red)
            plt.legend(loc = "best")
            plt.xlabel("Frames")
            plt.ylabel("log(Hz) / voiced and unvoiced flag")
            plt.show()

        else:
            plt.plot(standardization(features_map[ts], mean, std), label="%s" % (ts))
            plt.legend(loc = "best")
            plt.xlabel("Frames")
            plt.ylabel(ts)
            plt.show()
        



    


