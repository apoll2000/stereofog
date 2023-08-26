import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=15, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--num_test', type=int, default=50, help='number of test images')

args = parser.parse_args()

dataroot = args.dataroot
netG = args.netG
n_epochs = args.n_epochs
n_epochs_decay = args.n_epochs_decay
num_test = args.num_test

# List of dropout rates to test
dropout_rates = [0.1, 0.3, 0.5]

# List of commands to run
commands = []

# Generate the commands
for rate in dropout_rates:
    commands.append(
        f"python train.py --dataroot {dataroot} --name hyperparameter_dropout_{rate} --model pix2pix --direction BtoA --display_id -1 "
        f"--n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --netG {netG} --dropout_rate {rate}"
    )

# Run the commands
for command in commands:
    print("Running command:", command)
    subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for rate in dropout_rates:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_dropout_{rate} --model pix2pix"
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_dropout'}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for dropout rate completed.")