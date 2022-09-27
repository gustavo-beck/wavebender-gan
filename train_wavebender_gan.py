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
from wavebender_cgan import WaveDiscriminator, ResidualBlockGan
from wavebenderloss import WaveBenderLoss
from wavebender_ae_extension import ConvAutoencoder
from data_utils import FeaturesMelAugmentSelectionLoader, FeaturesMelSelectionCollate, FeaturesMelLoader, FeaturesMelSelectionAnyCollate
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from apex import amp
# Flexible integration for any Python script
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)
CB91_Blue = '#2CBDFE'
CB91_Red = '#DA6F6F'
CB91_Green = 'springgreen'

def save_wavebendernet(model, filepath):
    torch.save(model.state_dict(), filepath)

def test_generator(wavebender, generator, criterion_g, test_set, device):
    avg_loss = []
    with torch.no_grad():
        for batch_idx, (inputs_batch, targets_batch) in enumerate(test_set):

            # Predict Mel-Spectrograms
            mel = wavebender(inputs_batch.to(device))
            mel = mel.unsqueeze(1)

            # Clean Noisy Mel-Spectrogram with Enconder
            new_mel = generator(mel)

            # Compute Loss
            targets_batch = targets_batch.unsqueeze(1)
            loss = criterion_g(new_mel, targets_batch.to(device))

            # Append to average
            avg_loss.append(loss.item())
        
    # Return average loss
    return np.mean(avg_loss)

def least_square_gan(prediction = None, label = None):
    return 0.5 * torch.mean((prediction-label)**2)

def save_wavebendergan(discriminator, generator, d_path, g_path):
    torch.save(discriminator.state_dict(), d_path)
    torch.save(generator.state_dict(), g_path)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
    """
    Load input files of shape N x 11 x t*
    N: The number of train/test/val samples
    11: The number of features (time-series)
    t*: The timeframe of each sample

    Load mel-spectrograms files of shape N x 80 x t*
    N: The number of train/test/val samples
    80: The number of channels of the mel-spectrograms
    t*: The timeframe of each sample
    """

    # Get data, data loaders and collate function ready
    features_selected = ["f1", "f2",
                    "f0_contour",
                    "spectral_centroid", "spectral_slope"
                    ]

    train_loader = FeaturesMelAugmentSelectionLoader(**train_config)
    test_loader = FeaturesMelLoader(params.test_input_path, params.test_target_path, "sorted_test.txt")
    collate_fn_selection = FeaturesMelSelectionCollate(params.n_frames_per_step, features_selected)
    collate_fn = FeaturesMelSelectionAnyCollate(params.n_frames_per_step, features_selected)

    train_set = DataLoader(train_loader, num_workers=10, shuffle=False,
                              sampler=None,
                              batch_size=params.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn_selection)

    test_set = DataLoader(test_loader, num_workers=10, shuffle=False,
                              sampler=None, pin_memory=False,
                              batch_size=params.batch_size,
                              drop_last=True, collate_fn=collate_fn)

    return train_set, test_set

def train(params):

    # 1. Start a W&B run
    wandb.init(project='wavebender', entity='gustavo-beck')

    # Set randomness
    set_seed(params.seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load WaveBenderNet
    try:
        wavebender = WaveBenderNet(ResidualBlock, params)
        wavebender.load_state_dict(torch.load(params.wavebender_path))
        wavebender.to(device)
        print("LOADED PRE-TRAINED WAVEBENDER: ", params.wavebender_path)
    except:
        wavebender = WaveBenderNet(ResidualBlock, params).to(device)
        print("TRAIN WAVEBENDER FROM SCRATCH")

    # Load cGAN
    try:
        discriminator = WaveDiscriminator(ResidualBlockGan, params)
        discriminator.load_state_dict(torch.load(params.discriminator_path))
        discriminator.to(device)
        print("LOADED PRE-TRAINED cGAN Discriminator: ", params.discriminator_path)
    except:
        discriminator = WaveDiscriminator(ResidualBlockGan, params)
        discriminator.to(device)
        print("TRAIN cGAN DISCRIMINATOR FROM SCRATCH")

    # Load cGAN
    try:
        generator = ConvAutoencoder(params)
        generator.load_state_dict(torch.load(params.generator_path))
        generator.to(device)
        print("LOADED PRE-TRAINED cGAN Generator: ", params.generator_path)
    except:
        generator = ConvAutoencoder(params)
        generator.to(device)
        print("TRAIN cGAN GENERATOR FROM SCRATCH")

    # Define Model Loss based on the Loss of WaveGlow
    criterion_g = WaveBenderLoss()

    print("FIXING WAVEBENDER")
    for param in wavebender.parameters():
        param.requires_grad = False

    optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                lr=params.learning_rate,
                                betas=(0.5, 0.999),
                                weight_decay=params.weight_decay
                                )
    
    '''optimizer_g = torch.optim.Adam(list(wavebender.parameters()) + list(generator.parameters()),
                                lr=params.learning_rate,
                                betas=(0.5, 0.999),
                                weight_decay=params.weight_decay
                                )'''

    optimizer_g = torch.optim.Adam(generator.parameters(),
                                lr=params.learning_rate,
                                betas=(0.5, 0.999),
                                weight_decay=params.weight_decay
                                )

    # Prepare Training/Validation/Testing sets (Features and Mel-Spectrograms)
    train_set, test_set = prepare_data(params)

    # Declare Learning Curves
    discriminator_loss = []
    generator_loss = []
    wavebender_loss = []
    loss_test = []

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    mean = 0
    stddev = 0.01
    best_epoch = 0
    best_sig = np.inf

    # Compute training steps based on batch size and total epochs
    total_training_steps = int(np.ceil(11700 / params.batch_size) * params.total_training_epochs)

    # Compute warmup steps based on the amount of total training (e.g. 5% of them)
    warmup_steps = int(0.05 * total_training_steps)

    # Schedule the update of the learning rate
    #print("CYCLIC SCHEDULER ON")
    scheduler_d = LambdaLR(optimizer=optimizer_d, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps, total_training_steps))
    scheduler_g = LambdaLR(optimizer=optimizer_g, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps, total_training_steps))

    # Begin Training
    for epoch in tqdm(range(params.total_training_epochs)):
        print('TRAINING epoch', epoch + 1)
        wavebender_loss_tmp = []
        generator_loss_tmp = []
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_set):
            
            ###########################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            discriminator.zero_grad()
            ## Train with real images
            label = torch.full((params.batch_size,), real_label, dtype=torch.float, device=device)
            
            # Forward pass real batch through D
            targets_batch = targets_batch.unsqueeze(1)
            output_real = discriminator(targets_batch.to(device)).view(-1)

            # Calculate loss on real images
            errD_real = 0.5 * torch.sum((output_real - label) ** 2) / params.batch_size

            # Calculate gradients for Descriminator in backward pass
            # Accumulate D real loss
            errD_real.backward()
            D_x = output_real.mean().item()

            ## Train with fake images
            # Generate fake image batch with Generator Wavebender
            mel = wavebender(inputs_batch.to(device))
            mel = mel.unsqueeze(1)

            # Compute wavebender loss
            #trackW = torch.nn.MSELoss()(mel, targets_batch.to(device))

            # Generate fake image batch with G
            label.fill_(fake_label)
            
            # Add noise to the input
            noise = torch.autograd.Variable(mel.data.new(mel.size()).normal_(mean, stddev))
            fake = mel + noise

            # Compute Mel based on AutoEncoder
            # Clean Noisy Mel-Spectrogram with Enconder
            fake = generator(fake)

            # Compute wavebender loss
            errW = criterion_g(fake, targets_batch.to(device))

            # Classify all fake batch with D
            output_fake = discriminator(fake.detach()).view(-1)

            # Calculate D's loss on the fake images
            errD_fake = 0.5 * torch.sum((output_fake - label) ** 2) / params.batch_size

            # Calculate the gradients for this batch
            # Accumulate D fake loss
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update D
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), params.max_clip)

            # Update optimizers (Adam -> Model)
            optimizer_d.step()
            optimizer_d.zero_grad()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)

            # Calculate G's loss based on this output
            errG = 0.5 * torch.sum((output - label) ** 2) / params.batch_size
            
            errGTotal = errW

            # Accumulate G loss
            errGTotal.backward()

            D_G_z2 = output.mean().item()
            # Update G
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), params.max_clip)

            # Update optimizers (Adam -> Model)
            optimizer_g.step()
            optimizer_g.zero_grad()

            # Output training stats
            if batch_idx % 25 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\terrW: %.4f'
                    % (epoch, params.total_training_epochs, batch_idx, len(train_set),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, errW.item()))

            discriminator_loss.append(errD.item())
            generator_loss.append(errGTotal.item())
            generator_loss_tmp.append(errGTotal.item())
            wavebender_loss_tmp.append(errW.item())
            
            # Update learning rates
            scheduler_d.step()
            scheduler_g.step()
            wandb.log({"Discriminator Loss": errD.item(),
                       "Wavebender GAN Loss": errG.item(),
                       "Wavebender Net Loss": errW.item(),
                       "Total Generator Loss": errGTotal.item(),
                      }
                     )

        # Compute mean loss over all batches
        wavebender_loss.append(np.mean(wavebender_loss_tmp))
        test_adv_loss = np.mean(generator_loss_tmp)

        # Compute test loss
        loss_test.append(test_generator(wavebender, generator, criterion_g, test_set, device).item())
        
        # Check result
        print("CURRENT TEST LOSS", test_adv_loss)
        print("CURRENT TEST XSigmoid LOSS", loss_test[-1])
        if test_adv_loss < best_sig:
            best_sig = test_adv_loss
            best_epoch = epoch + 1
            print("NEW BEST: ", best_sig)
            print("NEW EPOCH: ", best_epoch)
            # Save model
            directory = "generator_checkpoints/"
            netD = "discriminator"
            netG = "wavebender_gan"
            save_wavebendergan(discriminator, generator, directory + netD + "_final_" + str(best_epoch) + ".pt", directory + netG + "_final_" + str(best_epoch) + ".pt")
            #save_wavebendernet(wavebender, "wavebender_gan_final_" + str(best_epoch) + ".pt")

        print("CURRENT BEST: ", best_sig)
        # Save images
        plt.imshow(fake.cpu().detach().numpy()[0, 0,:,:],interpolation='none',cmap=plt.cm.jet,origin='lower')
        fname = "generator_images/fake_short_augment_selection" + str(epoch + 1)
        plt.savefig(fname + ".png")
        plt.close()
        torch.save(fake.cpu().detach().numpy()[0, 0,:,:], fname + ".pt")
        if epoch == 0:
            plt.imshow(targets_batch.cpu().detach().numpy()[0, 0,:,:],interpolation='none',cmap=plt.cm.jet,origin='lower')
            fname = "generator_images/real_short_augment_selection"
            plt.savefig(fname + ".png")
            plt.close()
            torch.save(targets_batch.cpu().detach().numpy()[0, 0,:,:], fname + ".pt")

        # Print AVG Wavebender Loss
        print("AVG Wavebender Loss", np.mean(wavebender_loss_tmp))

    # Plot Losses
    episodes_range = range(len(discriminator_loss))
    plt.plot(episodes_range, discriminator_loss, label="Discriminator Loss", c=CB91_Red)
    plt.plot(episodes_range, generator_loss, label="Generator Loss", c=CB91_Blue)
    plt.xlabel("Episodes")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig("WavebenderGan_Disc_Gen_final_augment_6f_last.png")
    plt.close()

    epoch_range = range(len(wavebender_loss))
    plt.plot(epoch_range, wavebender_loss, label="Wavebender Train Loss", c=CB91_Blue)
    plt.plot(epoch_range, loss_test, label="Wavebender Test Loss", c=CB91_Green)
    plt.xlabel("Episodes")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig("WavebenderGan_final_augment_6f_last.png")
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
    parser.add_argument("--wavebender_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="wavebender_net.pt")
    parser.add_argument("--generator_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="wavebender_gan.pt")
    parser.add_argument("--discriminator_path", 
                        help="Define the wavebender model path",
                        type=str,
                        default="discriminator.pt")
    parser.add_argument("--val_input_path", 
                        help="Define the directory to read the features of val files",
                        type=str,
                        default="wavebender_features_data/val/")
    parser.add_argument("--train_target_path", 
                        help="Define the directory to read the mel train files",
                        type=str,
                        default="tacotron2_mel_spectrograms/train/")
    parser.add_argument("--test_target_path", 
                        help="Define the directory to read the mel test files",
                        type=str,
                        default="tacotron2_mel_spectrograms/test/")
    parser.add_argument("--val_target_path", 
                        help="Define the directory to read the mel val files",
                        type=str,
                        default="tacotron2_mel_spectrograms/val/")
    parser.add_argument("--seed", 
                        help="Define seed to shuffle dataset",
                        type=int,
                        default=1337)
    parser.add_argument("--learning_rate", 
                        help="Define learning rate",
                        type=float,
                        default= 1e-8)
    parser.add_argument("--weight_decay", 
                        help="Define weight_decay",
                        type=float,
                        default= 1e-4)
    parser.add_argument("--total_training_epochs", 
                        help="Define total amount of epochs, i.e. times of all training batch have been seen",
                        type=int,
                        default= 10)
    parser.add_argument("--n_frames_per_step", 
                        help="Currently each frame corresponts to one step",
                        type=int,
                        default= 1)
    parser.add_argument("--batch_size", 
                        help="How many samples in one batch",
                        type=int,
                        default= 2)
    parser.add_argument("--window_size", 
                        help="Window size to traverse the time-series",
                        type=int,
                        default= 32)
    parser.add_argument("--n_channels_in", 
                        help="How many inputs we pass through",
                        type=int,
                        default= 6)
    parser.add_argument("--n_channels_out", 
                        help="How many channels to predict",
                        type=int,
                        default= 80)
    parser.add_argument("--save_model", 
                        help="Define the directory to save WaveBenderNet",
                        type=str,
                        default="checkpoints/")
    parser.add_argument("--max_clip", 
                        help="Define the maximum/minimum clip absolute value",
                        type=int,
                        default = 5)
    parser.add_argument("-as", "--accumulation_steps", default=8, type=int)
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