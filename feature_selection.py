import argparse
import json
import os
import  random
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from data_utils import FeaturesMelLoader, FramesCollate
from wavebenderloss import WaveBenderLoss
from mlp import MLPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from apex import amp
import wandb

CB91_Blue = '#2CBDFE'
CB91_Red = '#DA6F6F'
CB91_Green = 'springgreen'

def save_mlp(model, filepath):
    torch.save(model.state_dict(), filepath)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def plot_performance(title, x_label, y_label, x_data, y_data, color=None):
    plt.plot(x_data, y_data, label=title, c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

def cyclic_learning_rate_with_warmup(warmup_steps, total_training_steps):
    def scheduler_function(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)

        else:  # Once you surpass the number of warmup steps,
            # Decay they learning rate close zero in a cosine manner
            x = np.cos(7. / 16. * np.pi * ((step - warmup_steps) / (total_training_steps - warmup_steps)))
            return x

    # Update learning rate scheduler
    return scheduler_function

def test_mlp(model, criterion, data, device):
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


def prepare_data(params, features_selected):

    # Get data, data loaders and collate function ready
    train_loader = FeaturesMelLoader(params.train_input_path, params.train_target_path, "sorted_train.txt")
    test_loader = FeaturesMelLoader(params.test_input_path, params.test_target_path, "sorted_test.txt")
    collate_fn = FramesCollate(features_selected)

    train_set = DataLoader(train_loader, num_workers=16, shuffle=False,
                              sampler=None,
                              batch_size=params.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    test_set = DataLoader(test_loader, num_workers=16, shuffle=False,
                              sampler=None, pin_memory=False,
                              batch_size=params.batch_size,
                              drop_last=True, collate_fn=collate_fn)

    return train_set, test_set

def train(params):
    # 1. Start a W&B run
    wandb.init(project='wavebender', entity='gustavo-beck')

    best_epoch = 0
    best_sig = np.inf

    # Set randomness
    set_seed(params.seed)

    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define Model Loss
    criterion = WaveBenderLoss()

    # Prepare Training/Testing sets (Features and Mel-Spectrograms)
    # Get global data stats (mean, std, max, min)
    with open(os.path.join("global_stats.json")) as f:
        global_stats = json.load(f)
    features = list(global_stats.keys())
    features_selected = ["f1", "f2", "f0_contour", "intensity", "spectral_slope", "spectral_centroid"]

    train_set, test_set = prepare_data(params, features_selected)
    num_features = len(features_selected) + 1
    output_dim = 80 # Mel-Spectrograms channels
    model = MLPModel(num_features, output_dim, dropout=0.20, n_hid=128)
    model.to(device)

     # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=params.learning_rate,
                                weight_decay=params.weight_decay
                                )
    
    # Compute training steps based on batch size and total epochs
    total_training_steps = int(np.ceil(11700 / params.batch_size) * params.total_training_epochs)

    # Compute warmup steps based on the amount of total training (e.g. 5% of them)
    warmup_steps = int(0.05 * total_training_steps)

    # Schedule the update of the learning rate
    print("CYCLIC SCHEDULER ON")
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps, total_training_steps))

    model.train()
    test_loss = []
    train_loss = []
    for epoch in tqdm(range(params.total_training_epochs)):
        print('TRAINING epoch', epoch + 1)
        train_loss_tmp = []

        # Batch training
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_set):

            # Set gradients to zero
            model.zero_grad()

            # Predict Mel-Spectrograms
            predictions = model(inputs_batch.to(device)) # 64 x 80

            # Compute Loss
            loss = criterion(predictions, targets_batch.to(device))

            # Store learning
            train_loss_tmp.append(loss.item())

            # Back-propagate loss through WaveBenderNet
            loss.backward()

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_clip)

            # Update optimizers (Adam -> Model)
            optimizer.step()

            # Set gradients to zero
            model.zero_grad()

            # Update Learning rate
            scheduler.step()

            wandb.log({
                       "XSigmoid train loss": loss
                      }
                     )

        # Compute mean loss over all batches
        train_loss.append(np.mean(train_loss_tmp))

        # Compute test loss
        test_loss.append(test_mlp(model, criterion, test_set, device).item())
        print("CURRENT TEST LOSS", test_loss[-1])
        # Check result
        if test_loss[-1] < best_sig:
            best_sig = test_loss[-1]
            best_epoch = epoch + 1
            print("NEW BEST: ", best_sig)
            print("NEW EPOCH: ", best_epoch)
            save_mlp(model, "mlp.pt")
        else:
            print("CURRENT TRAIN LOSS: ", train_loss[-1])
    
    epoch_range = range(len(train_loss))
    plt.plot(epoch_range, train_loss, label="Wavebender Train Loss", c=CB91_Blue)
    plt.plot(epoch_range, test_loss, label="Wavebender Test Loss", c=CB91_Green)
    plt.xlabel("Episodes")
    plt.ylabel("XSigmoid")
    plt.legend()
    plt.title("XSigmoid Loss")
    plt.suptitle("Features : %s, %s, %s, %s, %s, %s and %s" % (features_selected[0], features_selected[1], features_selected[2],
                                                    features_selected[3], features_selected[4], features_selected[5], "f0_mask"), fontsize = 6)
    plt.savefig("MLP_training_curve.png")
    plt.close()

    print("NEW BEST: ", best_sig)
    print("NEW EPOCH: ", best_epoch)

if __name__ == "__main__":
    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_path", 
                        help="Define the directory to read the the features of train files",
                        type=str,
                        default="wavebender_features_data/train/")
    parser.add_argument("--test_input_path", 
                        help="Define the directory to read the the features of test files",
                        type=str,
                        default="wavebender_features_data/test/")
    parser.add_argument("--train_target_path", 
                        help="Define the directory to read the mel train files",
                        type=str,
                        default="tacotron2_mel_spectrograms/train/")
    parser.add_argument("--test_target_path", 
                        help="Define the directory to read the mel test files",
                        type=str,
                        default="tacotron2_mel_spectrograms/test/")
    parser.add_argument("--seed", 
                        help="Define seed to shuffle dataset",
                        type=int,
                        default=1337)
    parser.add_argument("--learning_rate", 
                        help="Define learning rate",
                        type=float,
                        default= 1e-3)
    parser.add_argument("--weight_decay", 
                        help="Define weigth decay",
                        type=float,
                        default= 1e-4)
    parser.add_argument("--momentum", 
                        help="Define momentum",
                        type=float,
                        default=0.9)
    parser.add_argument("--nesterov", 
                        help="Define Nesterov",
                        type=bool,
                        default=True)
    parser.add_argument("--total_training_epochs", 
                        help="Define total amount of epochs, i.e. times of all training batch have been seen",
                        type=int,
                        default= 50)
    parser.add_argument("--batch_size", 
                        help="How many samples in one batch",
                        type=int,
                        default= 64)
    parser.add_argument("--save_model", 
                        help="Define the directory to save WaveBenderNet",
                        type=str,
                        default="checkpoints/")
    parser.add_argument("--max_clip", 
                        help="Define the maximum/minimum clip absolute value",
                        type=int,
                        default= 50)

    args = parser.parse_args()

    train(args)