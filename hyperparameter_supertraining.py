# Hyperparameter supertraining script
# This script will run the specified hyperparameter training scripts in sequence, in order to be able to run everything everywhere all at once.
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=15, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--num_test', type=int, default=50, help='number of test images')

args = parser.parse_args()

dataroot = args.dataroot
n_epochs = args.n_epochs
n_epochs_decay = args.n_epochs_decay
num_test = args.num_test

scripts = [
    "hyperparameter_DropoutRate.py",
    "hyperparameter_NetDmode.py"
    "hyperparameter_NetGmode.py",
    "hyperparameter_GANmode.py",
    "hyperparameter_ngf_ndf.py"
]

for script in scripts:
    print("Running hyperparameter script:", script)
    subprocess.call(f"python {script} --dataroot {dataroot} --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --num_test {num_test}", shell=True)