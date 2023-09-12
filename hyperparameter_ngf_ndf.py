import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=15, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--num_test', type=int, default=50, help='number of test images')
parser.add_argument('--test_only', action='store_true', help='if specified, skip training and perform testing only')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()

dataroot = args.dataroot
n_epochs = args.n_epochs
n_epochs_decay = args.n_epochs_decay
num_test = args.num_test
test_only = args.test_only
gpu_ids = args.gpu_ids

n_gen_filters = [32, 64, 128]

n_discrim_filters = [32, 64, 128]

checkpoints_dir = "checkpoints/hyperparameters/hyperparameter_ngf_ndf"

# List of experiments to run
experiments = []

# Test various widths of the U-Net (i.e., number of filters in the last conv layer)
for ngf in n_gen_filters:
    experiments.append(f"python train.py --dataroot {dataroot} --name hyperparameter_ngf_{ngf} --model pix2pix "
                       f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --ngf {ngf} --checkpoints_dir {checkpoints_dir}")

# Test various depths of the U-Net (i.e., number of discrim filters in the first conv layer)
for ndf in n_discrim_filters:
    experiments.append(f"python train.py --dataroot {dataroot} --name hyperparameter_ndf_{ndf} --model pix2pix "
                       f"--direction BtoA --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --ndf {ndf} --checkpoints_dir {checkpoints_dir}")


# Loop through experiments and execute each one sequentially
if not test_only:
    for command in experiments:
        subprocess.call(command, shell=True)

# Testing script
test_script = []

# Generate the test commands
for ngf in n_gen_filters:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_ngf_{ngf} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_ngf_ndf'} --ngf {ngf} --gpu_ids {gpu_ids} --checkpoints_dir {checkpoints_dir}"
    )

for ndf in n_discrim_filters:
    test_script.append(
        f"python test.py --dataroot {dataroot} --name hyperparameter_ndf_{ndf} --model pix2pix "
        f"--direction BtoA --num_test {num_test} --epoch latest --results_dir {f'results/hyperparameters/hyperparameter_ngf_ndf'} --ndf {ndf} --gpu_ids {gpu_ids} --checkpoints_dir {checkpoints_dir}"
    )

# Run the test commands
for command in test_script:
    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All hyperparameter tests for ngf and ndf completed.")