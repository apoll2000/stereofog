import subprocess

# List of experiments to run
experiments = []

# Test various depths of the U-Net (i.e., number of downsamplings)
for num_downs in [8, 9, 10]:
    experiments.append(f"python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_depth_{num_downs} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_{num_downs}")

# Test various widths of the U-Net (i.e., number of filters in the last conv layer)
for ngf in [64, 128, 256, 512]:
    experiments.append(f"python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_width_{ngf} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_256 --ngf {ngf}")

# Loop through experiments and execute each one sequentially
for command in experiments:
    subprocess.call(command, shell=True)

