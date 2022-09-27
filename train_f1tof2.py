import argparse
import json
import os
import  random
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from wavebendernet import WaveBenderNet, ResidualBlock
from wavebenderloss import WaveBenderLoss
from data_utils import F1AndF2Loader, F1AndF2Collate
from tqdm import tqdm
import matplotlib.pyplot as plt
from test_model import testing
import wandb

CB91_Blue = '#2CBDFE'
CB91_Green = 'springgreen'
CB91_Red = '#DA6F6F'

def save_wavebendernet(model, filepath):
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

def prepare_data(params):

    features_selected = ["f1", "f2"]
    # Get data, data loaders and collate function ready
    train_loader = F1AndF2Loader(params.train_input_path, "sorted_train.txt")
    test_loader = F1AndF2Loader(params.test_input_path, "sorted_test.txt")
    collate_fn = F1AndF2Collate(params.n_frames_per_step, features_selected)

    train_set = DataLoader(train_loader, num_workers=10, shuffle=False,
                              sampler=None,
                              batch_size=params.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    test_set = DataLoader(test_loader, num_workers=10, shuffle=False,
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

    # Load WaveBenderNet
    try:
        model = WaveBenderNet(ResidualBlock, params)
        model.load_state_dict(torch.load(params.wavebender_path))
        model.to(device)
        print("LOADED PRE-TRAINED WAVEBENDER")
    except:
        model = WaveBenderNet(ResidualBlock, params)
        model.to(device)
        print("TRAIN WAVEBENDER FROM SCRATCH")

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=params.learning_rate,
                                weight_decay=params.weight_decay
                                )
    
    # Define Model Loss
    criterion = WaveBenderLoss()

    # Prepare Training/Testing sets (Features and Mel-Spectrograms)
    train_set, test_set = prepare_data(params)

    # Compute training steps based on batch size and total epochs
    total_training_steps = int(np.ceil(11700 / params.batch_size) * params.total_training_epochs)

    # Compute warmup steps based on the amount of total training (e.g. 5% of them)
    warmup_steps = int(0.05 * total_training_steps)

    # Schedule the update of the learning rate
    print("CYCLIC SCHEDULER ON")
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps, total_training_steps))

    # Declare Learning Curves
    train_loss = []
    test_loss = []

    # Begin Training
    model.train()
    for epoch in tqdm(range(params.total_training_epochs)):
        print('TRAINING epoch', epoch + 1)
        train_loss_tmp = []

        # Batch training
        for batch_idx, (inputs_batch, targets_batch) in tqdm(enumerate(train_set)):
    
            # Set gradients to zero
            model.zero_grad()

            # Predict Mel-Spectrograms
            predictions = model(inputs_batch.to(device))

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

            # Print learning loss
            if (batch_idx + 1) % 10 == 0:
                print("CURRENT LOSS: ", loss)

            wandb.log({
                       "F1 to F2 Loss": loss
                      }
                     )

        # Compute mean loss over all batches
        train_loss.append(np.mean(train_loss_tmp))

        # Check result
        if train_loss[-1] < best_sig:
            best_sig = train_loss[-1]
            best_epoch = epoch + 1
            save_wavebendernet(model, "f1tof2.pt")

        # Compute test loss
        t_loss = testing(model, criterion, test_set, device)
        test_loss.append(t_loss.item())

        print("Train loss: ", train_loss[-1])
        print("Test loss: ", test_loss[-1])

    # Plot Losses
    epoch_range = range(params.total_training_epochs)
    plot_performance('Training Loss', 'Epochs', 'Loss', epoch_range, train_loss, CB91_Blue)
    plot_performance('Testing Loss', 'Epochs', 'Loss', epoch_range, test_loss, CB91_Green)
    plt.savefig("F1_to_F2_Learning_Curves.png")
    plt.close()

    # Print the best epoch and train loss
    print("BEST EPOCH: ", best_epoch)
    print("BEST LOSS: ", best_sig)
    

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
    parser.add_argument("--wavebender_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="f1tof2.pt")
    parser.add_argument("--learning_rate", 
                        help="Define learning rate",
                        type=float,
                        default= 1e-5)
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
                        default= 20)
    parser.add_argument("--n_frames_per_step", 
                        help="Currently each frame corresponts to one step",
                        type=int,
                        default= 1)
    parser.add_argument("--batch_size", 
                        help="How many samples in one batch",
                        type=int,
                        default= 16)
    parser.add_argument("--window_size", 
                        help="Window size to traverse the time-series",
                        type=int,
                        default= 32)
    parser.add_argument("--n_channels_in", 
                        help="How many inputs we pass through",
                        type=int,
                        default= 1)
    parser.add_argument("--n_channels_out", 
                        help="How many channels to predict",
                        type=int,
                        default= 1)
    parser.add_argument("--save_model", 
                        help="Define the directory to save WaveBenderNet",
                        type=str,
                        default="checkpoints/")
    parser.add_argument("--max_clip", 
                        help="Define the maximum/minimum clip absolute value",
                        type=int,
                        default= 50)
    parser.add_argument("-as", "--accumulation_steps", default=2, type=int)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration',
                        default="config_wavebender_augment.json")

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    global train_config
    train_config = config["train_config"]

    train(args)