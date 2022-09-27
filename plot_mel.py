import matplotlib.pyplot as plt
import torch
import os
import re
import json

with open(os.path.join("wavebender_features_data/test/sorted_test.txt")) as f:
    matches = re.findall(r"'(.+?)'", f.read())

mels_path = "tacotron2_mel_spectrograms/test/"
features_path = "wavebender_features_data/test/" 
print(len(matches))
for file in matches:
    mel = torch.load(os.path.join(mels_path + "mel_" + file + ".pt"))
    with open(os.path.join(features_path + file + ".json")) as f:
        feat = json.load(f)
    print(len(feat[0]))
    print(mel.shape)
    plt.imshow(mel,interpolation='none',cmap=plt.cm.jet,origin='lower')
    plt.show()
    break
    