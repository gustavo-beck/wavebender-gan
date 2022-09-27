from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
from tqdm import tqdm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import and concatenate all features from all samples
path = "wavebender_features_data/train/"
with open(path + "sorted_train.txt") as f:
    files = re.findall('"([^"]*)"', f.read())
sorted_files = files[::-1]
print("LOADED FILES NAMES")

features_names = [
                "f1", "f2", "f3", "f4", "log_energy",
                "zero_crossing_rate", "f0_counter",
                "intensity", "spectral_centroid",
                "spectral_slope", "spectral_spread",
                "f2-f1", "f3-f2", "f4-f3"
                ]

features_correlation = {}

for filename in tqdm(sorted_files):
    with open(os.path.join(path + filename + ".json")) as f:
        features = json.load(f)
    # Add f2-f1
    features.append([a - b for a, b in zip(features[1], features[0])])
    # Add f3-f2
    features.append([a - b for a, b in zip(features[2], features[1])])
    # Add f4-f3
    features.append([a - b for a, b in zip(features[3], features[2])])

    for idx, f in enumerate(features):
        if features_names[idx] not in features_correlation:
            features_correlation[features_names[idx]] = f
        else:
            features_correlation[features_names[idx]].extend(f)

print("LOADED FEATURES")
print("VECTOR SIZE: ", len(features_correlation["f1"]))

# Compute Spearman Rho coefficient and p
coeff_mat = np.zeros((len(features_names), len(features_names)))

for x, f_x in tqdm(enumerate(features_names)):
    for y, f_y in enumerate(features_names):
        if x == y:
            coeff_mat[x, y] = 1
        else:
            coeff, p = spearmanr(features_correlation[f_x], features_correlation[f_y])
            coeff_mat[x, y] = coeff
            print("CORRELATION %s BETWEEN %s IS %.4f" % (features_names[x], features_names[y], coeff))

ax = sns.heatmap(coeff_mat,
                annot=True,
                annot_kws={"size": 7},
                linewidths=0.5,
                linecolor='white',
                vmin=-1,
                vmax=1,
                xticklabels=features_names,
                yticklabels=features_names,
                cmap="Spectral")
plt.savefig("Features_Correlation.png", bbox_inches='tight')
plt.close()

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['Feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in tqdm(range(df.shape[1]))]
    return (vif)

# Construct VIF dataframe
original_features = pd.DataFrame.from_dict(features_correlation)
vif_table = calculate_vif(original_features)
vif_table = vif_table.sort_values(by = 'VIF', ascending = False, ignore_index = True)
print(vif_table)