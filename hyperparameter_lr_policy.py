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

checkpoints_dir = "checkpoints/hyperparameters/hyperparameter_lr_policy"

# List of discriminator architectures to test
lr_policies = ['linear', 'step', 'plateau', 'cosine']

# List of commands to run
commands = []

# Generate the commands
for lr_policy in lr_policies:
    commands.append(
        f"python train.py --dataroot {dataroot} --name hyperparameter_lr_policy_{lr_policy} --model pix2pix "
        f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --lr_policy {lr_policy} --checkpoints_dir {checkpoints_dir} --display_id 0"
    )

# Run the commands
for command in commands:
    print("Running command:", command)
    subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for netD in netD_architectures:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_lr_policy_{lr_policy} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_lr_policy'} --checkpoints_dir {checkpoints_dir}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for netD mode completed.")