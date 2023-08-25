import os
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--model_name', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--num_test', type=int, default=1, help='number of test images')

args = parser.parse_args()

dataroot = args.dataroot
model_name = args.model_name
num_test = args.num_test

# model_name = 'stereofog_pix2pix'
# dataroot = 'datasets/stereofog_images'

checkpoints_directory = f'checkpoints/{model_name}'




epochs = sorted(list(set([int(item.replace('_net_G.pth', '').replace('_net_D.pth', '')) for item in os.listdir(checkpoints_directory) if '.pth' in item and 'latest' not in item])))

for epoch in epochs:
    command = f'python test.py --dataroot {dataroot} --direction BtoA --model pix2pix --name {model_name} --results_dir {f"results/epochs/{model_name}_epochs" + str(epoch)} --epoch {epoch} --num_test {num_test}'

    print("Running command:", command)
    subprocess.call(command, shell=True)

print("All epochs tested.")