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
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
import seaborn as sns
from tqdm import tqdm
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import parselmouth
import sounddevice as sd
from sklearn.metrics import mean_squared_error
import torch
from wavebendernet import WaveBenderNet, ResidualBlock
from wavebender_ae_extension import ConvAutoencoder
from mel2samp import files_to_list
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

MAX_WAV_VALUE = 32768.0

def normalization(ts, minimum, maximum):
    norm = ts - minimum
    norm /= (maximum - minimum)
    return norm

def standardization(ts, mean, std):
    stand = ts - mean
    stand /= std
    return stand

def destandardization(ts, mean, std):
    destand = ts * std
    destand += mean
    return destand

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
    
class ParamObject(object):
    pass

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

def compute_mse(wave_original, features_selection, wavebender, generator, hifi, device, params, scale, position):
    # Time-series
    ts_features = compute_features(wave_original)

    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())

    # The time series must be normalized in order to enhance the performance of the WveBenderNet
    norm_ts = torch.FloatTensor(len(features_selection) + 1, len(ts_features[0])).zero_()
    original_norm_ts = torch.FloatTensor(len(features_selection), len(ts_features[0])).zero_()

    for idx, feature in enumerate(ts_features):
        # Normalize/Standardize values
        if features_selection[idx] == "f0_contour":
            # Generate mask
            feature_t = np.copy(feature)
            feature_t = torch.tensor(feature_t)
            mask_zeros = torch.zeros(feature_t.shape)
            mask_ones = torch.ones(feature_t.shape)
            f0_mask = torch.where(feature_t > 0, mask_ones, mask_zeros)
            norm_ts[-1] = f0_mask

            # Perform interpolation of log
            if features_selection[int(position)] == "f0_contour" and scale != 1.0:
                feature *= scale
            feature_l = np.copy(feature)
            feature_l = np.log(feature_l)
            feature_l[feature_l == -np.inf] = np.nan
            nans, x = np.isnan(feature_l), lambda z: z.nonzero()[0]
            feature_l[nans] = np.interp(x(nans), x(~nans), feature_l[~nans])
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            max_g = global_stats[features_selection[idx]]["max"]
            min_g = global_stats[features_selection[idx]]["min"]
            original_norm_ts[idx] = torch.FloatTensor(normalization(feature_l, min_g, max_g))
            norm_ts[idx] = torch.FloatTensor(standardization(feature_l, mean, std))
        else:
            mean = global_stats[features_selection[idx]]["mean"]
            std = global_stats[features_selection[idx]]["std"]
            max_g = global_stats[features_selection[idx]]["max"]
            min_g = global_stats[features_selection[idx]]["min"]
            if int(position) == idx and scale != 1.0:
                feature *= scale
                original_norm_ts[idx] = torch.FloatTensor(normalization(feature, min_g, max_g))
                norm_ts[idx] = torch.FloatTensor(standardization(feature, mean, std))
            else:
                original_norm_ts[idx] = torch.FloatTensor(normalization(feature, min_g, max_g))
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

        # Destandardize and Normalize
        mean = global_stats[features_selection[int(position + 1)]]["mean"]
        std = global_stats[features_selection[int(position + 1)]]["std"]
        max_g = global_stats[features_selection[int(position + 1)]]["max"]
        min_g = global_stats[features_selection[int(position + 1)]]["min"]
        original_norm_ts[int(position + 1)] = normalization(destandardization(norm_ts[int(position + 1)], mean, std), min_g, max_g)

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

        # Destandardize and Normalize
        mean = global_stats[features_selection[int(position - 1)]]["mean"]
        std = global_stats[features_selection[int(position - 1)]]["std"]
        max_g = global_stats[features_selection[int(position - 1)]]["max"]
        min_g = global_stats[features_selection[int(position - 1)]]["min"]
        original_norm_ts[int(position - 1)] = normalization(destandardization(norm_ts[int(position - 1)], mean, std), min_g, max_g)

    # Predict mel Wavebender
    mel_wavebender = wavebender(norm_ts.unsqueeze(0).to(device)).unsqueeze(1)

    # Enhance mel Encoder
    mel = generator(mel_wavebender)

    # Predict audio from wavebender + gan
    with torch.no_grad():
        y_g_hat = hifi(mel[0, 0,:,:].unsqueeze(0))
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        write("manipulated.wav", 22050, audio)

    # Get reconstructed time-series
    rec_wav = Waveform("manipulated.wav", sample_rate=22050)
    reconstruct_ts = compute_features(rec_wav)

    norm_rec_ts = []
    for idx, feature in enumerate(reconstruct_ts):
        max_g = global_stats[features_selection[idx]]["max"]
        min_g = global_stats[features_selection[idx]]["min"]
        if features_selection[idx] == "f0_contour":
            feature = np.log(feature)
            feature[feature == -np.inf] = np.nan
            nans, x = np.isnan(feature), lambda z: z.nonzero()[0]
            feature[nans] = np.interp(x(nans), x(~nans), feature[~nans])
            norm_rec_ts.append(normalization(feature, min_g, max_g))
        else:
            norm_rec_ts.append(normalization(feature, min_g, max_g))


    mse_list = {}

    for idx, (f, f_0) in enumerate(zip(norm_rec_ts, original_norm_ts.detach().numpy().tolist())):
        if idx < 3:
            mse_list[features_selection[idx]] = mean_squared_error(f, f_0)
            '''plt.plot(range(len(f)), f, label="Synthesized {} ".format(features_selection[idx]))
            plt.plot(range(len(f_0)), f_0, label="Original {}".format(features_selection[idx]))
            plt.legend()
            plt.savefig(features_selection[idx])
            plt.close()'''
        else:
            mse_list[features_selection[idx]] = mean_squared_error(f[:-1], f_0)
            '''plt.plot(range(len(f[:-1])), f[:-1], label="Synthesized {} ".format(features_selection[idx]))
            plt.plot(range(len(f_0)), f_0, label="Original {}".format(features_selection[idx]))
            plt.legend()
            plt.savefig(features_selection[idx])
            plt.close()'''

    return mse_list

def plot_features(ts, ts_o, f_list):

    mat_mse_aux = []

    for idx, (f, f_0) in enumerate(zip(ts, ts_o)):
        plt.plot(range(len(f)), f, label="Synthesized {} ".format(f_list[idx]))
        plt.plot(range(len(f_0)), f_0, label="Original {}".format(f_list[idx]))
        plt.legend()
        plt.show()
    
def read_audios(features_selection, scales, file_path, wavebender, generator, hifi, device, params):

    # Get files names
    waveforms = {}
    for s in scales:
        feature_map = {}
        for file in os.listdir(file_path):
            if file.endswith(".wav"):
                # Read audio
                wave_original = Waveform(os.path.join(file_path, file), sample_rate=22050)
                for position, f in enumerate(features_selection):
                    if str(s) in waveforms:
                        if str(f) in feature_map:
                            waveforms[str(s)][str(f)].append(compute_mse(wave_original, features_selection, wavebender, generator, hifi, device, params, s, position))
                        
                        else:
                            feature_map[str(f)] = 1
                            waveforms[str(s)][str(f)] = [compute_mse(wave_original, features_selection, wavebender, generator, hifi, device, params, s, position)]
                    else:
                        waveforms[str(s)] = {}
                        waveforms[str(s)][str(f)] = [compute_mse(wave_original, features_selection, wavebender, generator, hifi, device, params, s, position)]
                        feature_map[str(f)] = 1

    audio_files = list(waveforms.keys())

    return waveforms

def plot_box_plot_scale(data):

    scales = list(data.keys())
    features = list(data[scales[0]].keys())
    print(scales)
    print(features)

    for f in features:
        data_mean = []
        for s in scales:
            data_mean.append(data[s][f])
        box = plt.boxplot(data_mean, labels = scales, vert=True, patch_artist=True)

        cm = plt.cm.get_cmap('rainbow')
        colors = [cm(val/len(data_mean)) for val in range(len(data_mean))]

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("{} manipulation reconstruction errors".format(f))
        plt.ylabel("Relative MSE")
        plt.xlabel("Scales")
        plt.tight_layout()
        plt.ylim([0, 0.025])
        plt.savefig(f + "_manipulation_error.png")
        plt.close()

    return

def plot_box_plot_feature(data):

    scales = list(data.keys())
    features = list(data[scales[0]].keys())
    print(scales)
    print(features)
    
    for s in scales:
        for f in features:
            f1 = []
            f2 = []
            f0 = []
            centroid = []
            slope = []
            for audio in range(len(data[s][f])):
                f1.append(data[s][f][audio][features[0]])
                f2.append(data[s][f][audio][features[1]])
                f0.append(data[s][f][audio][features[2]])
                centroid.append(data[s][f][audio][features[3]])
                slope.append(data[s][f][audio][features[4]])

            data_mean = [f1,
                        f2,
                        f0,
                        centroid,
                        slope
                        ]

            box = plt.boxplot(data_mean, labels = features, vert=False, patch_artist=True)

            cm = plt.cm.get_cmap('rainbow')
            colors = [cm(val/len(data_mean)) for val in range(len(data_mean))]

            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            plt.title("Low-level signal properties \n when manipulating {}".format(f))
            plt.ylabel("Relative MSE")
            plt.xlabel("Features")
            plt.tight_layout()
            plt.xlim([0, 0.07])
            plt.savefig(f + "_manipulation_error_over_features.png")
            plt.close()

    return

def plot_heatmap(data):

    scales = list(data.keys())
    features = list(data[scales[0]].keys())
    print(scales)
    print(features)
    data_mean = []
    data_std = []
    for s in scales:
        for f in features:
            f1 = []
            f2 = []
            f0 = []
            centroid = []
            slope = []
            for audio in range(len(data[s][f])):
                f1.append(data[s][f][audio][features[0]])
                f2.append(data[s][f][audio][features[1]])
                f0.append(data[s][f][audio][features[2]])
                centroid.append(data[s][f][audio][features[3]])
                slope.append(data[s][f][audio][features[4]])

            data_mean.append([np.mean(f1),
                        np.mean(f2),
                        np.mean(f0),
                        np.mean(centroid),
                        np.mean(slope)
                        ])

            data_std.append([np.std(f1),
                        np.std(f2),
                        np.std(f0),
                        np.std(centroid),
                        np.std(slope)
                        ])

    ax = sns.heatmap(np.array(data_mean).T, annot=True, xticklabels=features, yticklabels=features)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top') 
    plt.xlabel("Manipulated feature with scale 1.3")
    plt.ylabel("Relative MSE")
    plt.xticks(rotation=90)
    plt.savefig("Heatmap.png", bbox_inches='tight')
    plt.close()
            
    return

def analyze(params):

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


    features_selection = ["f1", "f2",
                    "f0_contour",
                    "spectral_centroid", "spectral_slope"
                    ]

    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    print("Compute MSE")
    manipulation_performance = read_audios(features_selection, scales, params.file_name, wavebender, generator, hifi, device, params)
    # manipulation_performance = {'1.3': {'f1': [{'f1': 0.04579905176592532, 'f2': 0.009196428375483741, 'f0_contour': 0.008934821783148096, 'spectral_centroid': 0.0009434532954304722, 'spectral_slope': 0.00010213378466852827}, {'f1': 0.04769096429728757, 'f2': 0.00936439335024911, 'f0_contour': 0.003941265908577693, 'spectral_centroid': 0.0007724923467036863, 'spectral_slope': 0.00012538296594709485}, {'f1': 0.06599441365779815, 'f2': 0.013307404657507373, 'f0_contour': 0.0035814343415428588, 'spectral_centroid': 0.0017984573763831966, 'spectral_slope': 0.0004488465014715071}, {'f1': 0.04784117315283579, 'f2': 0.007660677587152366, 'f0_contour': 0.00946852229779608, 'spectral_centroid': 0.000636174173754199, 'spectral_slope': 9.227002960853853e-05}, {'f1': 0.036656397017076695, 'f2': 0.00945573469892365, 'f0_contour': 0.001142894664741692, 'spectral_centroid': 0.0008266340847352798, 'spectral_slope': 0.0001719651608731312}, {'f1': 0.04927736151692512, 'f2': 0.011130666960136893, 'f0_contour': 0.004087708931629175, 'spectral_centroid': 0.0009269826053903602, 'spectral_slope': 0.00032261209110063496}, {'f1': 0.05433560860940924, 'f2': 0.010316317350389716, 'f0_contour': 0.0001967898188316373, 'spectral_centroid': 0.0007093290629152525, 'spectral_slope': 0.00011173701644074437}, {'f1': 0.051736381993576705, 'f2': 0.010150869893071136, 'f0_contour': 0.0017493273958385889, 'spectral_centroid': 0.0007527723917189245, 'spectral_slope': 0.00016159813895269534}, {'f1': 0.04660346383958973, 'f2': 0.008995741730355632, 'f0_contour': 0.004137708197290738, 'spectral_centroid': 0.0007917276451978144, 'spectral_slope': 0.00011418273345079553}, {'f1': 0.06267869204241809, 'f2': 0.0089523330447985, 'f0_contour': 0.00039462953360696023, 'spectral_centroid': 0.000827474759318568, 'spectral_slope': 0.00017337354165787133}], 'f2': [{'f1': 0.03782122433965743, 'f2': 0.03689623385892726, 'f0_contour': 0.003674437909884603, 'spectral_centroid': 0.001789644439393314, 'spectral_slope': 0.0002158202268374506}, {'f1': 0.04236639699766885, 'f2': 0.0338655401643115, 'f0_contour': 0.015313829659146735, 'spectral_centroid': 0.001365904172478739, 'spectral_slope': 0.0002467376535773851}, {'f1': 0.05166405252092711, 'f2': 0.04185642558609594, 'f0_contour': 0.0007314704314144998, 'spectral_centroid': 0.004374302576036344, 'spectral_slope': 0.0006632863262376086}, {'f1': 0.05080533822122647, 'f2': 0.040946059616072755, 'f0_contour': 0.008253875872885709, 'spectral_centroid': 0.0013076086764042564, 'spectral_slope': 6.396970142523175e-05}, {'f1': 0.036992468875727075, 'f2': 0.05021959585973553, 'f0_contour': 0.0011985761297455663, 'spectral_centroid': 0.001960014494728694, 'spectral_slope': 0.0002848319509549977}, {'f1': 0.04099840430103749, 'f2': 0.043078728728300035, 'f0_contour': 0.0020836681030648872, 'spectral_centroid': 0.0017692515971096748, 'spectral_slope': 0.0004224682538461277}, {'f1': 0.0474614195066469, 'f2': 0.0518379032979853, 'f0_contour': 0.0002404307932793551, 'spectral_centroid': 0.0013309207786513797, 'spectral_slope': 4.561120345977122e-05}, {'f1': 0.040540651508206085, 'f2': 0.044393645261275726, 'f0_contour': 0.0005101850335747317, 'spectral_centroid': 0.002087122331744657, 'spectral_slope': 0.00025150111718839804}, {'f1': 0.03515542498169648, 'f2': 0.03279918748643163, 'f0_contour': 0.005581446184813379, 'spectral_centroid': 0.0014386619808608324, 'spectral_slope': 0.0001638080300294622}, {'f1': 0.04334431911839651, 'f2': 0.03939214163531992, 'f0_contour': 0.0011906986230951027, 'spectral_centroid': 0.0014268422243715426, 'spectral_slope': 6.29286527137347e-05}], 'f0_contour': [{'f1': 0.033220034657211475, 'f2': 0.013972861340942317, 'f0_contour': 0.0022389566472582607, 'spectral_centroid': 0.0009519941086409089, 'spectral_slope': 0.00010028661734725961}, {'f1': 0.023280911162748996, 'f2': 0.01335512402481292, 'f0_contour': 0.0017217670185293394, 'spectral_centroid': 0.0006432275837824869, 'spectral_slope': 0.000274847701588929}, {'f1': 0.042497305088122064, 'f2': 0.017624167164276002, 'f0_contour': 0.0004392773486033025, 'spectral_centroid': 0.002247518430924427, 'spectral_slope': 0.0003613077583044817}, {'f1': 0.029351668597864145, 'f2': 0.010363621153107822, 'f0_contour': 0.009970499998601823, 'spectral_centroid': 0.0004907572034009017, 'spectral_slope': 9.445927274892742e-05}, {'f1': 0.018971967884352917, 'f2': 0.013988899909987819, 'f0_contour': 0.006114008157312211, 'spectral_centroid': 0.0006951698802164384, 'spectral_slope': 0.00015830641061520038}, {'f1': 0.03267352215249731, 'f2': 0.01592271073824439, 'f0_contour': 0.009947587323824888, 'spectral_centroid': 0.0007832535067556571, 'spectral_slope': 0.0002365547773212469}, {'f1': 0.02308709035002885, 'f2': 0.017111505251099002, 'f0_contour': 0.00022614665187461355, 'spectral_centroid': 0.0005119463763490167, 'spectral_slope': 4.459442731678498e-05}, {'f1': 0.027942810517153995, 'f2': 0.017327252016865205, 'f0_contour': 0.0006936922130957145, 'spectral_centroid': 0.000670382216031481, 'spectral_slope': 0.000189716782295666}, {'f1': 0.02729048606899612, 'f2': 0.013864414072827742, 'f0_contour': 0.0016361006158529875, 'spectral_centroid': 0.0005295767965188246, 'spectral_slope': 8.485947035914324e-05}, {'f1': 0.03161000942154539, 'f2': 0.016031805500453675, 'f0_contour': 0.005137373592047545, 'spectral_centroid': 0.0006524303588643144, 'spectral_slope': 0.00010373000617919669}], 'spectral_centroid': [{'f1': 0.033410629360600505, 'f2': 0.014322096100632911, 'f0_contour': 0.0027128458308537527, 'spectral_centroid': 0.006949446192113678, 'spectral_slope': 0.0003873461513772207}, {'f1': 0.03954372280438214, 'f2': 0.012379605178993672, 'f0_contour': 0.0023341673688860366, 'spectral_centroid': 0.004027114080988856, 'spectral_slope': 0.00029550032981242225}, {'f1': 0.04435020317569718, 'f2': 0.01832825052890454, 'f0_contour': 0.0004564442089686046, 'spectral_centroid': 0.017528319670682867, 'spectral_slope': 0.001236349098231605}, {'f1': 0.03236853184117883, 'f2': 0.013004127698115474, 'f0_contour': 0.005129658542354239, 'spectral_centroid': 0.002492808992548173, 'spectral_slope': 6.845571832690282e-05}, {'f1': 0.03238839964156555, 'f2': 0.02377839062358301, 'f0_contour': 0.0007516144355044387, 'spectral_centroid': 0.009733038572301735, 'spectral_slope': 0.0005712013927188044}, {'f1': 0.03644131989392903, 'f2': 0.01796245318586585, 'f0_contour': 0.004963590822882282, 'spectral_centroid': 0.011090843001059779, 'spectral_slope': 0.0006269049360222989}, {'f1': 0.025948888333727202, 'f2': 0.020231402629931915, 'f0_contour': 0.000491314671211022, 'spectral_centroid': 0.0030182901231563287, 'spectral_slope': 0.00011913825371428633}, {'f1': 0.028952856147072704, 'f2': 0.02157992022814562, 'f0_contour': 0.004451192593692484, 'spectral_centroid': 0.006342639512100296, 'spectral_slope': 0.0002260639850654065}, {'f1': 0.033658605187004795, 'f2': 0.01649672271287725, 'f0_contour': 0.002055084570498982, 'spectral_centroid': 0.0023975690780405185, 'spectral_slope': 0.00017712542347394724}, {'f1': 0.036081343350580816, 'f2': 0.019396486504319948, 'f0_contour': 0.0021269574095456032, 'spectral_centroid': 0.003774906707466012, 'spectral_slope': 0.00012130469214119817}], 'spectral_slope': [{'f1': 0.027487924674950606, 'f2': 0.015992594537844667, 'f0_contour': 0.0032362123477935717, 'spectral_centroid': 0.0008189001884223801, 'spectral_slope': 0.00018724386184201474}, {'f1': 0.023672191228520763, 'f2': 0.01132307922221839, 'f0_contour': 0.0010574727655879874, 'spectral_centroid': 0.000741142521683213, 'spectral_slope': 0.0005269447440257035}, {'f1': 0.04573370075551075, 'f2': 0.01587717253212633, 'f0_contour': 0.008231590084857842, 'spectral_centroid': 0.0017309823332230375, 'spectral_slope': 0.0007289141582440276}, {'f1': 0.021505232934890917, 'f2': 0.00977742899748303, 'f0_contour': 0.004368782496910954, 'spectral_centroid': 0.0004772079265224329, 'spectral_slope': 9.138981827837869e-05}, {'f1': 0.016215012881498077, 'f2': 0.01263381731852382, 'f0_contour': 0.00037966162506162333, 'spectral_centroid': 0.0006829356532668827, 'spectral_slope': 0.00030282854937682295}, {'f1': 0.03056667979683789, 'f2': 0.015202237174379922, 'f0_contour': 0.0018357773168893225, 'spectral_centroid': 0.0008035794919733853, 'spectral_slope': 0.00040916369413199286}, {'f1': 0.02064215992084042, 'f2': 0.019285553085037592, 'f0_contour': 0.00018359328107715777, 'spectral_centroid': 0.0005715888695140513, 'spectral_slope': 5.671104425684664e-05}, {'f1': 0.02436798882472776, 'f2': 0.016389560928733712, 'f0_contour': 0.0008269026324398924, 'spectral_centroid': 0.0005535330879880031, 'spectral_slope': 0.00019768028068388913}, {'f1': 0.02796631073460954, 'f2': 0.014238624266784617, 'f0_contour': 0.007720282475356589, 'spectral_centroid': 0.0005422036131653851, 'spectral_slope': 0.0001847637096200278}, {'f1': 0.031312349107025794, 'f2': 0.019430700568061173, 'f0_contour': 0.007677225687702803, 'spectral_centroid': 0.0005627284896099935, 'spectral_slope': 0.00010004608265987723}]}}
    with open('mse_wavebender_gan_manipulation.txt', 'w') as f:
        f.write(json.dumps(str(manipulation_performance)))

    # Plot manipulation reconstruction error
    print("Plot Box Analysis per feature")
    print(manipulation_performance)
    # plot_box_plot_feature(manipulation_performance)
    # plot_heatmap(manipulation_performance)
    plot_box_plot_scale(manipulation_performance)

if __name__ == "__main__":

    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--file_name", 
                        help="Define the directory to read the audios",
                        type=str,
                        default="manipulated_speech/")
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
    parser.add_argument('--config_file', default="hifi/config.json")
    parser.add_argument('--hifi_file', default="hifi/hifi_model")

    args = parser.parse_args()

    with open(args.config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    analyze(args)


