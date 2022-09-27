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
import torch

# Read .txt files
def read_files(input_path):
    with open(input_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files

# Extract time-series and continuos features
def ext_features(file_path):
    # Get .wav file
    wave = Waveform(path=os.path.join(file_path), sample_rate=22050)
    
    # Time-series
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    log_energy = wave.log_energy_slidingwindow()[0] # Easier to interpret than the signal 
    zcr = wave.zerocrossing_slidingwindow()[0] # Can be interpreted as a measure of the noisiness of a signal
    f0_contour = wave.f0_contour(method='rapt', f0_max=1000)[0] # The number of glottal pulses in a second
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

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    time_series = [ts.tolist() for ts in ts_features]

    return time_series

if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory', default="wavebender_features_data/")

    args = parser.parse_args()

    # Read files
    test_files = read_files("test_files.txt")

    time_series_features = [
        "f1", "f2", "f3", "f4", "log_energy",
        "zero_crossing_rate", "f0_contour",
        "intensity", "spectral_centroid",
        "spectral_slope", "spectral_spread"
    ]

    waveform_features = {}
    length_test_dist = []
    length_train_dist = []
    length_dist = []

    # Extract test features of the waveforms
    for file in tqdm(test_files):
        # Extract features
        features = ext_features(file)
        length_test_dist.append(len(features[0]))
        file_name = file.partition("wavs/")[2]

        # Store features
        with open(args.output_dir + "test/" + file_name + ".json", 'w') as outfile:
            json.dump(features, outfile)

        for i, f in enumerate(time_series_features):
            if f not in waveform_features:
                waveform_features[f] = features[i]
            else:
                waveform_features[f].extend(features[i])

    # Store test length histogram
    length_dist.extend(length_test_dist)
    plt.hist(length_test_dist, color = 'blue', edgecolor = 'black', bins=32)
    plt.title('Histogram of Test Audio Length')
    plt.xlabel('Audio Lengths')
    plt.ylabel('Audios')
    plt.savefig("Histogram_test.png")
    plt.close()

    # Store sorted files based on length
    sorted_test_idx = sorted(range(len(length_test_dist)), key=lambda k: length_test_dist[k])
    sorted_test = []
    for idx in tqdm(sorted_test_idx):
        sorted_test.append(str(test_files[idx].partition("wavs/")[2]))
    print("Sorted Test Txt Size: ", len(sorted_test))
    with open(os.path.join(args.output_dir + "test/sorted_test.txt"), 'w') as outfile:
        json.dump(sorted_test, outfile)

    
    # Read files
    train_files = read_files("train_files.txt")

    # Extract train features of the waveforms
    for file in tqdm(train_files):
        # Extract features
        features = ext_features(file)
        length_train_dist.append(len(features[0]))
        file_name = file.partition("wavs/")[2]
        
        # Store features
        with open(args.output_dir + "train/" + file_name + ".json", 'w') as outfile:
            json.dump(features, outfile)

        for i, f in enumerate(time_series_features):
            if f not in waveform_features:
                waveform_features[f] = features[i]
            else:
                waveform_features[f].extend(features[i])

    # Store train length histogram
    plt.hist(length_train_dist, color = 'blue', edgecolor = 'black', bins=32)
    plt.title('Histogram of Train Audio Length')
    plt.xlabel('Audio Lengths')
    plt.ylabel('Audios')
    plt.savefig("Histogram_train.png")
    plt.close()

    # Store sorted files based on length
    sorted_train_idx = sorted(range(len(length_train_dist)), key=lambda k: length_train_dist[k])
    sorted_train = []
    for idx in tqdm(sorted_train_idx):
        sorted_train.append(str(train_files[idx].partition("wavs/")[2]))
    print("Sorted Train Txt Size: ", len(sorted_train))

    with open(os.path.join(args.output_dir + "train/sorted_train.txt"), 'w') as outfile:
        json.dump(sorted_train, outfile)

    # Store all length histogram
    length_dist.extend(length_train_dist)
    plt.hist(length_dist, color = 'blue', edgecolor = 'black', bins=32)
    plt.title('Histogram of All Audio Length')
    plt.xlabel('Audio Lengths')
    plt.ylabel('Audios')
    plt.savefig("Histogram_all.png")
    plt.close()

    # Compute Global stats  
    data_stats = {}
    for feature in time_series_features:
        if feature == "f0_contour":
            entire_time_series = torch.log(torch.tensor(waveform_features[feature]))
            entire_time_series[entire_time_series == -np.inf] = np.nan
            entire_time_series = entire_time_series.numpy()
            nans, x = np.isnan(entire_time_series), lambda z: z.nonzero()[0]
            entire_time_series[nans] = np.interp(x(nans), x(~nans), entire_time_series[~nans])
        else:
            entire_time_series = waveform_features[feature]
        data_stats[feature] = {}
        data_stats[feature]["mean"] = str(np.mean(entire_time_series))
        data_stats[feature]["max"] = str(np.max(entire_time_series))
        data_stats[feature]["min"] = str(np.min(entire_time_series))
        data_stats[feature]["std"] = str(np.std(entire_time_series))
    with open("test_stats.json", 'w') as outfile:
        json.dump(data_stats, outfile)
        



    


