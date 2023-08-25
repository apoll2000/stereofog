import subprocess

# Training

commands = [
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_1 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_9blocks",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_2 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_6blocks",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_3 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_256",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_4 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_9blocks --norm instance",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_5 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_6blocks --norm instance",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_6 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_256 --norm instance",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_7 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_9blocks --norm none",
    "python train.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_8 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 15 --save_epoch_freq 5 --netG resnet_6blocks --norm none"
]

# Loop through commands and execute each one sequentially
for command in commands:
    subprocess.call(command, shell=True)

# Testing

commands = [
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_1 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_9blocks",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_2 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_6blocks",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_3 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG unet_256",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_4 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_9blocks --norm instance",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_5 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_6blocks --norm instance",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_6 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG unet_256 --norm instance",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_7 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_9blocks --norm none",
    "python test.py --dataroot ./datasets/20230714_minlap_700crop_pro_pix2pix --name experiment_8 --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100 --epoch latest --netG resnet_6blocks --norm none"
]

# Loop through commands and execute each one sequentially
for command in commands:
    subprocess.call(command, shell=True)