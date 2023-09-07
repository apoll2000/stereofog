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

checkpoints_dir = "checkpoints/hyperparameters/hyperparameter_init_type"

# List of discriminator architectures to test
init_types = ['normal', 'xavier', 'kaiming', 'orthogonal']

# List of commands to run
commands = []

# Generate the commands
for init_type in init_types:
    commands.append(
        f"python train.py --dataroot {dataroot} --name hyperparameter_init_type_{init_type} --model pix2pix "
        f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --init_type {init_type} --checkpoints_dir {checkpoints_dir} --display_id 0"
    )

# Run the commands
for command in commands:
    print("Running command:", command)
    subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for init_type in init_types:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_init_type_{init_type} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_init_type'} --init_type {init_type} --checkpoints_dir {checkpoints_dir}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for init type completed.")