import random
import numpy as np
import json.scanner
import torch
import torch.utils.data
import os
import re
from os import listdir
from os.path import isfile, join
from scipy.io.wavfile import read
import sys
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import parselmouth
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

def standardization(ts, mean, std):
    # Normal Z-score
    z_score = ts - mean
    z_score /= std

    return z_score

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    wav = np.copy(data)
    return torch.FloatTensor(wav)

def features_sample(file_path, audio_start, seg_len, sr):
    wave_entire = Waveform(path=os.path.join(file_path), sample_rate=sr)
    wave = Waveform(signal=wave_entire.waveform[audio_start: audio_start+seg_len], sample_rate=wave_entire.sample_rate)

    # Time-series
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    log_energy = wave.log_energy_slidingwindow()[0] # Easier to interpret than the signal 
    zcr = wave.zerocrossing_slidingwindow()[0] # Can be interpreted as a measure of the noisiness of a signal
    f0_contour = wave.f0_contour(method='rapt', f0_max=8000)[0] # The number of glottal pulses in a second
    intensity = wave.intensity()[0] # Power carried by sound waves per unit area.
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source
    spectral_spread = wave.spectral_spread()[0] # Associated with the bandwidth of the signal. E.g. Individual tonal sounds with isolated peaks will result in a low spectral spread.

    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())

    ts_features = [f1, f2, f3, f4,
                    log_energy, zcr,
                    f0_contour, intensity,
                    spectral_centroid, spectral_slope,
                    spectral_spread
                    ]

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    norm_ts = torch.FloatTensor(len(ts_features), len(ts_features[0])).zero_()

    for idx, ts in enumerate(ts_features):
        mean = global_stats[features[idx]]["mean"]
        std = global_stats[features[idx]]["std"]
        norm_ts[idx] = torch.FloatTensor(standardization(ts, mean, std))

    return torch.FloatTensor(norm_ts)

def features_augmentation(wave):

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
    ts_tensor = torch.FloatTensor(len(ts_features), len(ts_features[0])).zero_()

    for idx, ts in enumerate(ts_features):
        ts_tensor[idx] = torch.FloatTensor(ts)

    return torch.FloatTensor(ts_tensor)

def features_augmentation_selection(wave):

    # Time-series
    formants = wave.formants_slidingwindow()  # Concentration of acoustic energy around a particular frequency in the speech wave.
    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    f0_contour = wave.f0_contour(method='rapt', f0_max=1000)[0] # The number of glottal pulses in a second
    # intensity = wave.intensity()[0] # Power carried by sound waves per unit area.
    spectral_centroid = wave.spectral_centroid()[0] # It determines the brightness of a sound.
    spectral_slope = wave.spectral_slope()[0] # Related to the nature of the sound source

    ts_features = [f1, f2,
                    f0_contour,
                    spectral_centroid, spectral_slope
                    ]

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    ts_tensor = torch.FloatTensor(len(ts_features), len(ts_features[0])).zero_()

    for idx, ts in enumerate(ts_features):
        ts_tensor[idx] = torch.FloatTensor(ts)

    return torch.FloatTensor(ts_tensor)

class FeaturesMelLoader(torch.utils.data.Dataset):

    def __init__(self, features_path, mels_path, sorted_path):
        self.features_path = features_path
        self.mels_path = mels_path
        with open(os.path.join(self.features_path + sorted_path)) as f:
            self.sorted_files = re.findall('"([^"]*)"', f.read())
        self.sorted_files = self.sorted_files[::-1]

    def get_mel_features_pair(self, file_idx):
        # separate filename and features
        features = self.get_features(file_idx)
        mel = self.load_mel(file_idx)

        return (features, mel)

    def load_mel(self, filename):
        mel = torch.load(os.path.join(self.mels_path + "mel_" + filename + ".pt"))
        return mel

    def get_features(self, filename):
        with open(os.path.join(self.features_path + filename + ".json")) as f:
            features = json.load(f)
        return features

    def __getitem__(self, index):
        return self.get_mel_features_pair(self.sorted_files[index])

    def __len__(self):
        return len(self.sorted_files)


class FeaturesMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step # Currently only 1
    
    def __call__(self, batch):
        """
        Collate's training batch from normalized features and mel-spectrogram
        PARAMS
        ------
        batch: [features_normalized, mel_normalized]
        """
        # Right zero-pad all features sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0][0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        min_input_length = input_lengths[-1]

        # Define mask for time-series
        features_padded = torch.stack([torch.FloatTensor(len(batch[0][0]), max_input_len).zero_() for i in range(len(batch))])

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())

        # Fill zeros to masks
        for i in range(len(ids_sorted_decreasing)):
            features_ts = batch[ids_sorted_decreasing[i]][0]
            for idx, feature in enumerate(features_ts):
                # Normalize/Standardize values
                mean = global_stats[features[idx]]["mean"]
                std = global_stats[features[idx]]["std"]
                features_padded[i][idx][:torch.tensor(feature).size(0)] = torch.FloatTensor(standardization(np.array(feature), mean, std))

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel

        return features_padded, mel_padded

class FeaturesMelAudioLoader(torch.utils.data.Dataset):

    def __init__(self, files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        random.seed(1234)
        with open(os.path.join(files)) as f:
            self.audio_files = re.findall('"([^"]*)"', f.read())
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio = load_wav_to_torch("data/wavs/" + filename)

        # Take segment
        if audio.size(0) >= self.segment_length:
            audio_std = 0
            while audio_std < 1e-5:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                segment = audio[audio_start:audio_start+self.segment_length]
                audio_std = segment.std()
            audio = segment
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE
        features = features_sample("data/wavs/" + filename,
                                audio_start = audio_start,
                                seg_len = self.segment_length,
                                sr = self.sampling_rate
                                )

        return (features, mel, audio)

    def __len__(self):
        return len(self.audio_files)

class FeaturesMelAugmentLoader(torch.utils.data.Dataset):

    def __init__(self, files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        random.seed(1234)
        with open(os.path.join(files)) as f:
            self.audio_files = re.findall('"([^"]*)"', f.read())
        self.audio_files = self.audio_files[::-1]
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_augmented_mel(self, audio):
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        sound = parselmouth.Sound(os.path.join("data/wavs/" + filename))
        factor = random.uniform(0.80, 1.20)
        manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
        parselmouth.praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
        parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
        sound_octave_up = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")

        # Get augmented audio, mel and features
        wave = Waveform(signal=sound_octave_up.values[0], sample_rate= self.sampling_rate)
        wav = np.copy(wave.waveform)
        mel = self.get_augmented_mel(torch.FloatTensor(wav))
        features = features_augmentation(wave)

        return (features, mel)

    def __len__(self):
        return len(self.audio_files)

class FeaturesMelAugmentSelectionLoader(torch.utils.data.Dataset):

    def __init__(self, files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        random.seed(1234)
        with open(os.path.join(files)) as f:
            self.audio_files = re.findall('"([^"]*)"', f.read())
        self.audio_files = self.audio_files[::-1]
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_augmented_mel(self, audio):
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        sound = parselmouth.Sound(os.path.join("data/wavs/" + filename))
        factor = random.uniform(0.80, 1.20)
        manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
        parselmouth.praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
        parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
        sound_octave_up = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")

        # Get augmented audio, mel and features
        intensity_scale = random.uniform(np.log(0.1), 0)
        wave = Waveform(signal=sound_octave_up.values[0] * np.exp(intensity_scale), sample_rate= self.sampling_rate)
        wav = np.copy(wave.waveform)
        mel = self.get_augmented_mel(torch.FloatTensor(wav))
        features = features_augmentation_selection(wave)

        return (features, mel)

    def __len__(self):
        return len(self.audio_files)

class FeaturesInferenceLoader(torch.utils.data.Dataset):

    def __init__(self, features_path, sorted_files):
        self.features_path = features_path
        self.sorted_files = sorted_files
        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            self.global_stats = json.load(f)
        self.features = list(self.global_stats.keys())

    def get_features(self, filename):
        with open(os.path.join(self.features_path + filename + ".json")) as f:
            features_ts = json.load(f)
        
        for idx, feature in enumerate(features_ts):
            # Normalize/Standardize values
            mean = self.global_stats[self.features[idx]]["mean"]
            std = self.global_stats[self.features[idx]]["std"]
            features_ts[idx] = standardization(np.array(feature), mean, std)

        return features_ts

    def __getitem__(self, index):
        return torch.FloatTensor(self.get_features(self.sorted_files[index]))

    def __len__(self):
        return len(self.sorted_files)

class FramesCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, features_selection):
        self.features_selection = features_selection
    def __call__(self, batch):

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())
        features_indeces = [features.index(f) for f in self.features_selection]
        features_concat = {}
        
        for idx, (feature_set, mel) in enumerate(batch):
            if idx == 0:
                mels = mel
            else:
                mels = torch.hstack((mels, mel))
            for idy in features_indeces:
                feature = feature_set[idy]
                # Normalize/Standardize values
                mean = global_stats[features[idy]]["mean"]
                std = global_stats[features[idy]]["std"]
                feature = torch.tensor(feature)
                if features[idy] == "f0_contour":
                    mask_zeros = torch.zeros(feature.shape)
                    mask_ones = torch.ones(feature.shape)
                    f0_mask = torch.where(feature > 0, mask_ones, mask_zeros)
                    if "f0_mask" not in features_concat:
                        features_concat["f0_mask"] = f0_mask
                    else:
                        features_concat["f0_mask"] = torch.hstack((features_concat["f0_mask"], f0_mask))
                    # Perform interpolation of log
                    feature = torch.log(feature)
                    feature[feature == -np.inf] = np.nan
                    feature = feature.numpy()
                    nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
                    feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
                    feature_std  = torch.FloatTensor(standardization(feature, mean, std))
                else:
                    feature_std  = torch.FloatTensor(standardization(feature, mean, std))
                if features[idy] not in features_concat:
                    features_concat[features[idy]] = feature_std
                else:
                    features_concat[features[idy]] = torch.hstack((features_concat[features[idy]], feature_std))
            
        length = len(features_concat[features[0]])
        batch_size = 64
        channel_size = 80
        random_indeces = random.sample(list(np.arange(length)), batch_size)
        f_input = torch.zeros((len(list(features_concat.keys())), batch_size))
        f_output = torch.zeros((channel_size, batch_size))
        
        for idx, f in enumerate(list(features_concat.keys())):
            f_input[idx] = features_concat[f][random_indeces]

        f_output = mels[:, random_indeces]
        return f_input.T, f_output.T

class FeaturesMelSelectionCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, features_selection):
        self.n_frames_per_step = n_frames_per_step # Currently only 1
        self.features_selection = features_selection
    def __call__(self, batch):
        """
        Collate's training batch from normalized features and mel-spectrogram
        PARAMS
        ------
        batch: [features_normalized, mel_normalized]
        """
        # Right zero-pad all features sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0][0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        min_input_length = input_lengths[-1]

        # Define mask for time-series
        features_padded = torch.stack([torch.FloatTensor(len(self.features_selection) + 1, max_input_len).zero_() for i in range(len(batch))])

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())
        features_indeces = [features.index(f) for f in self.features_selection]

        # Fill zeros to masks
        for i in range(len(ids_sorted_decreasing)):
            features_ts = batch[ids_sorted_decreasing[i]][0]
            for idx, feature in enumerate(features_ts):
                length = feature.size(0)
                # Normalize/Standardize values
                if self.features_selection[idx] == "f0_contour":
                    # Generate mask
                    mask_zeros = torch.zeros(feature.shape)
                    mask_ones = torch.ones(feature.shape)
                    f0_mask = torch.where(feature > 0, mask_ones, mask_zeros)
                    features_padded[i][-1][:feature.size(0)] = f0_mask

                    # Perform interpolation of log
                    feature = torch.log(feature)
                    feature[feature == -np.inf] = np.nan
                    feature = feature.numpy()
                    nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
                    feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
                    mean = global_stats[self.features_selection[idx]]["mean"]
                    std = global_stats[self.features_selection[idx]]["std"]
                    features_padded[i][idx][:length] = torch.FloatTensor(standardization(feature, mean, std))
                else:
                    mean = global_stats[self.features_selection[idx]]["mean"]
                    std = global_stats[self.features_selection[idx]]["std"]
                    features_padded[i][idx][:length] = torch.FloatTensor(standardization(feature, mean, std))

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel

        return features_padded, mel_padded

class FeaturesMelSelectionAnyCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, features_selection):
        self.n_frames_per_step = n_frames_per_step # Currently only 1
        self.features_selection = features_selection
    def __call__(self, batch):
        """
        Collate's training batch from normalized features and mel-spectrogram
        PARAMS
        ------
        batch: [features_normalized, mel_normalized]
        """
        # Right zero-pad all features sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0][0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        min_input_length = input_lengths[-1]

        # Define mask for time-series
        features_padded = torch.stack([torch.FloatTensor(len(self.features_selection) + 1, max_input_len).zero_() for i in range(len(batch))])

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())
        features_indeces = [features.index(f) for f in self.features_selection]

        # Fill zeros to masks
        for i in range(len(ids_sorted_decreasing)):
            features_ts = batch[ids_sorted_decreasing[i]][0]
            aux = 0
            for idx, feature in enumerate(features_ts):
                feature = torch.tensor(feature)
                length = feature.size(0)
                # Normalize/Standardize values
                if idx in features_indeces:
                    if self.features_selection[aux] == "f0_contour":
                        # Generate mask
                        mask_zeros = torch.zeros(feature.shape)
                        mask_ones = torch.ones(feature.shape)
                        f0_mask = torch.where(feature > 0, mask_ones, mask_zeros)
                        features_padded[i][-1][:feature.size(0)] = f0_mask

                        # Perform interpolation of log
                        feature = torch.log(feature)
                        feature[feature == -np.inf] = np.nan
                        feature = feature.numpy()
                        nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
                        feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
                        mean = global_stats[self.features_selection[aux]]["mean"]
                        std = global_stats[self.features_selection[aux]]["std"]
                        features_padded[i][aux][:length] = torch.FloatTensor(standardization(feature, mean, std))
                    else:
                        mean = global_stats[self.features_selection[aux]]["mean"]
                        std = global_stats[self.features_selection[aux]]["std"]
                        features_padded[i][aux][:length] = torch.FloatTensor(standardization(feature, mean, std))
                    aux += 1

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel

        return features_padded, mel_padded

class F1AndF2Loader(torch.utils.data.Dataset):

    def __init__(self, features_path, sorted_path):
        self.features_path = features_path
        with open(os.path.join(self.features_path + sorted_path)) as f:
            self.sorted_files = re.findall('"([^"]*)"', f.read())
        self.sorted_files = self.sorted_files[::-1]

    def get_features_pair(self, file_idx):
        # separate filename and features
        features = self.get_features(file_idx)

        return features

    def get_features(self, filename):
        with open(os.path.join(self.features_path + filename + ".json")) as f:
            features = json.load(f)
        return features

    def __getitem__(self, index):
        return self.get_features_pair(self.sorted_files[index])

    def __len__(self):
        return len(self.sorted_files)

class F1AndF2Collate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, features_selection):
        self.n_frames_per_step = n_frames_per_step # Currently only 1
        self.features_selection = features_selection
    def __call__(self, batch):
        # Right zero-pad all features sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        min_input_length = input_lengths[-1]

        # Define mask for time-series
        features_padded = torch.stack([torch.FloatTensor(len(self.features_selection), max_input_len).zero_() for i in range(len(batch))])

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())
        features_indeces = [features.index(f) for f in self.features_selection]

        # Fill zeros to masks
        for i in range(len(ids_sorted_decreasing)):
            features_ts = batch[ids_sorted_decreasing[i]]
            aux = 0
            for idx, feature in enumerate(features_ts):
                feature = torch.tensor(feature)
                length = feature.shape[0]
                # Normalize/Standardize values
                if idx in features_indeces:
                    mean = global_stats[self.features_selection[aux]]["mean"]
                    std = global_stats[self.features_selection[aux]]["std"]
                    features_padded[i][aux][:length] = torch.FloatTensor(standardization(feature, mean, std))
                    aux += 1

        return features_padded[:, 0].unsqueeze(1), features_padded[:, 1].unsqueeze(1)

class F2AndF1Collate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, features_selection):
        self.n_frames_per_step = n_frames_per_step # Currently only 1
        self.features_selection = features_selection
    def __call__(self, batch):
        # Right zero-pad all features sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        min_input_length = input_lengths[-1]

        # Define mask for time-series
        features_padded = torch.stack([torch.FloatTensor(len(self.features_selection), max_input_len).zero_() for i in range(len(batch))])

        # Get global data stats (mean, std, max, min)
        with open(os.path.join("global_stats.json")) as f:
            global_stats = json.load(f)
        features = list(global_stats.keys())
        features_indeces = [features.index(f) for f in self.features_selection]

        # Fill zeros to masks
        for i in range(len(ids_sorted_decreasing)):
            features_ts = batch[ids_sorted_decreasing[i]]
            aux = 0
            for idx, feature in enumerate(features_ts):
                feature = torch.tensor(feature)
                length = feature.shape[0]
                # Normalize/Standardize values
                if idx in features_indeces:
                    mean = global_stats[self.features_selection[aux]]["mean"]
                    std = global_stats[self.features_selection[aux]]["std"]
                    features_padded[i][aux][:length] = torch.FloatTensor(standardization(feature, mean, std))
                    aux += 1
                    
        return features_padded[:, 1].unsqueeze(1), features_padded[:, 0].unsqueeze(1)