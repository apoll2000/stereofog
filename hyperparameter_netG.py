import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=15, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--test_only', action='store_true', help='if specified, skip training and perform testing only')
parser.add_argument('--num_test', type=int, default=50, help='number of test images')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()

dataroot = args.dataroot
n_epochs = args.n_epochs
n_epochs_decay = args.n_epochs_decay
test_only = args.test_only
num_test = args.num_test
gpu_ids = args.gpu_ids

checkpoints_dir = "checkpoints/hyperparameters/hyperparameter_netG"

# List of generator architectures to test
netG_archictectures = ['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128']

# List of commands to run
commands = []

# Generate the commands
for netG in netG_archictectures:
    commands.append(
        f"python train.py --dataroot {dataroot} --name hyperparameter_netG_{netG} --model pix2pix "
        f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --netG {netG} --checkpoints_dir {checkpoints_dir} --display_id 0"
    )

# Run the commands
if not test_only:
    for command in commands:
        print("Running command:", command)
        subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for netG in netG_archictectures:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_netG_{netG} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_netG'} --netG {netG} --gpu_ids {gpu_ids} --checkpoints_dir {checkpoints_dir}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for netG mode completed.")