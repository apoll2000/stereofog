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

# List of GAN modes to test
GAN_modes = ["vanilla", "lsgan", "wgangp"]

# List of commands to run
commands = []

# Generate the commands
for GAN_mode in GAN_modes:
    commands.append(
        f"python train.py --dataroot {dataroot} --name hyperparameter_GAN_mode_{GAN_mode} --model pix2pix "
        f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --gan_mode {GAN_mode}"
    )

# Run the commands
for command in commands:
    print("Running command:", command)
    subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for GAN_mode in GAN_modes:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_GAN_mode_{GAN_mode} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_GAN_mode_{netD}'}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for GAN mode completed.")