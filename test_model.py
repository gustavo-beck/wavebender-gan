import  random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def testing(model, criterion, data, device):
    avg_loss = []
    with torch.no_grad():
        for batch_idx, (inputs_batch, targets_batch) in enumerate(data):
            # Predict Mel-Spectrograms
            predictions = model(inputs_batch.to(device))

            # Compute Loss
            loss = criterion(predictions, targets_batch.to(device))

            # Append to average
            avg_loss.append(loss.item())
        
    # Return average loss
    return np.mean(avg_loss)

def testing_wavebender_glow(wavebender, criterion_wavebender, data, device):
    avg_loss = []
    with torch.no_grad():
        for batch_idx, (inputs_batch, targets_batch, audio) in enumerate(data):
            # Predict Mel-Spectrograms
            mel = wavebender(inputs_batch.to(device))

            # Compute Loss
            loss = criterion_wavebender(mel, targets_batch.to(device))

            # Append to average
            avg_loss.append(loss.item())


    # Return average losses
    return np.mean(avg_loss)