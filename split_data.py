# Split LJ Speech dataset
# 95% to train and 5% test
# The goal is to create 2 .txt files

from os import listdir
from os.path import isfile, join
import argparse
import random

def split_data(path, train_size, seed):
    # Define randomness
    random.seed(seed)

    # Read all files in directory
    files = [path + f for f in listdir(path) if isfile(join(path, f))]

    # Shuffle files
    random.shuffle(files)

    # Split train set
    data_size = len(files)
    train_data = files[:int(data_size * train_size)]
    test_data = files[int(data_size * train_size):]
    print("TRAIN SIZE: ", len(train_data))
    print("TEST SIZE: ", len(test_data))
    print("ALL SAMPLES", len(test_data) + len(train_data))

    return train_data, test_data

if __name__ == "__main__":
    # Get defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_path", 
                        help="Define the directory to read the .wav files",
                        type=str,
                        default="data/wavs/")
    parser.add_argument("--train_size", 
                        help="Define train size",
                        type=float,
                        default=0.95)
    parser.add_argument("--seed", 
                        help="Define seed to shuffle dataset",
                        type=int,
                        default=1337)
    args = parser.parse_args()

    # Split files into train, test, val files
    train, test = split_data(args.filelist_path, args.train_size, args.seed)

    # Store files
    with open('train_files.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    
    with open('test_files.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)




