# Script to evaluate a certain pix2pix hyperparameter

import argparse
import os
import random
import matplotlib.pyplot as plt
from utils_stereofog import calculate_model_results

parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (e.g., results/hyperparameters/hyperparameter_netG)')
parser.add_argument('--ratio', type=float, default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')
parser.add_argument('--num_images', type=int, default=5, help='number of images to plot for evaluation')
parser.add_argument('--shuffle', action='store_true', help='if specified, shuffle the images before plotting')
parser.add_argument('--seed', type=int, default=16, help='seed for the random shuffling of the images')
parser.add_argument('--flip', action='store_true', help='if specified, make the rows correspond to different test images, not different models (i.e., rows and columns are flipped)')

results_path = parser.parse_args().results_path
ratio = parser.parse_args().ratio
num_images = parser.parse_args().num_images
shuffle = parser.parse_args().shuffle
seed = parser.parse_args().seed
flip = parser.parse_args().flip

fontsize = 16

if 'final_models' in results_path:
    hyperparameter = 'final_models'
else:
    hyperparameter = results_path.split('/')[-1].replace('hyperparameter_', '')

subfolders = [item for item in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, item))]

num_models = len(subfolders)

if results_path.split('/')[-1].count('_') == 1:
    models = [item.replace(results_path.split('/')[-1] + '_', '') for item in subfolders]
else:
    models = [item.replace('hyperparameter_', '') for item in subfolders]
    sorted(models)

# Part that needs to be inserted due to the way pix2pix saves the images
subpath_addition = 'test_latest/images'

images = [item for item in os.listdir(os.path.join(results_path, subfolders[0], subpath_addition)) if 'fake_B' in item]

if shuffle:
    random.seed(seed)
    random.shuffle(images)

try:
    images_to_plot = images[:num_images]

except:
    print(f'Number of images to plot ({num_images}) is greater than the number of images available ({len(images)}). Defaulting to {len(images)} images.')
    images_to_plot = images


limit = len(images_to_plot)

# os.listdir('./datasets/stereofog_data/A/test')[0]
# orig_img = plt.imread('./datasets/stereofog_data/A/test/2023-08-01-04__20.bmp')
# ratio = orig_img.shape[1]/orig_img.shape[0]
width_per_image = 4
height_per_image = width_per_image / ratio

# ax = [fig.add_subplot(num_models+2,limit,i+1) for i in range(limit*(num_models+2))]

# ax[0].text(0.5, 1.05, 'fake', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[0].transAxes)
# ax[1].text(0.5, 1.05, 'foggy real', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[1].transAxes)
# ax[2].text(0.5, 1.05, 'clear real', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[2].transAxes)

# Calculating SSIM and CW-SSIM scores for each model
SSIM_scores = []
CW_SSIM_scores = []
Pearson_image_correlations = []

for model_index, model in enumerate(models):
    mean_SSIM, mean_CW_SSIM = calculate_model_results(os.path.join(results_path, subfolders[model_index]))
    SSIM_scores.append(mean_SSIM)
    CW_SSIM_scores.append(mean_CW_SSIM)


if flip:
    fig, ax = plt.subplots(limit, num_models+2, figsize=(width_per_image*(num_models+2), height_per_image*limit))

    ax[0, 0].text(0.5,1.1, 'fogged', transform=ax[0, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k')
    ax[0, 1].text(0.5,1.1, 'original', transform=ax[0, 1].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k')

    for i in range(limit):

        # Plotting the ground truth image
        img_original = plt.imread(os.path.join(results_path, subfolders[0], subpath_addition, images_to_plot[i][:-10] + 'real_A' + '.png'))
        ax[i, 0].imshow(img_original, aspect='auto')
        ax[i, 0].axis('off')

        # Plotting the fogged image
        img_fogged = plt.imread(os.path.join(results_path, subfolders[0], subpath_addition, images_to_plot[i][:-10] + 'real_B' + '.png'))
        ax[i, 1].imshow(img_fogged, aspect='auto')
        ax[i, 1].axis('off')

        # Plotting each of the model's results
        for j in range(num_models):
            img = plt.imread(os.path.join(results_path, subfolders[j], subpath_addition, images_to_plot[i]))
            ax[i, j+2].imshow(img, aspect='auto')
            if i == 0:
                ax[i, j+2].text(0.5,1.3, models[j], transform=ax[i, j+2].transAxes, horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k', zorder=20)
            ax[i, j+2].axis('off')

    # Plotting each model's scores
    for j in range(num_models):
        ax[0, j+2].text(0.5,1.12, f'SSIM:{SSIM_scores[j]:.2f}\nCW-SSIM:{CW_SSIM_scores[j]:.2f}', transform=ax[0, j+2].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='black', color='k')


else:
    fig, ax = plt.subplots(num_models+2, limit, figsize=(limit*width_per_image,(num_models+2)*height_per_image)) # num_models+2 to acommodate ground truth and fogged image

    ax[0, 0].text(-0.1,0.5, 'fogged', transform=ax[0, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k', rotation='vertical')
    ax[1, 0].text(-0.1,0.5, 'original', transform=ax[1, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k', rotation='vertical')

    for i in range(limit):

        # Plotting the ground truth image
        img_original = plt.imread(os.path.join(results_path, subfolders[0], subpath_addition, images_to_plot[i][:-10] + 'real_A' + '.png'))
        ax[0, i].imshow(img_original, aspect='auto')
        ax[0, i].axis('off')

        # Plotting the fogged image
        img_fogged = plt.imread(os.path.join(results_path, subfolders[0], subpath_addition, images_to_plot[i][:-10] + 'real_B' + '.png'))
        ax[1, i].imshow(img_fogged, aspect='auto')
        ax[1, i].axis('off')

        # Plotting each of the model's results
        for j in range(num_models):
            img = plt.imread(os.path.join(results_path, subfolders[j], subpath_addition, images_to_plot[i]))
            ax[j+2, i].imshow(img, aspect='auto')
            if i == 0:
                ax[j+2, i].text(-0.25,0.5, models[j], transform=ax[j+2, i].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='black', color='k', rotation='vertical')
            ax[j+2, i].axis('off')

    # Plotting each model's scores
    for j in range(num_models):
        ax[j+2, 0].text(-0.1,0.5, f'SSIM:{SSIM_scores[j]:.2f}\nCW-SSIM:{CW_SSIM_scores[j]:.2f}', transform=ax[j+2, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='black', color='k', rotation='vertical')

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(os.path.join(f"{results_path}", f"{hyperparameter}_evaluation.png"), bbox_inches='tight', pad_inches=0)

print(f"Saved evaluation figure to {os.path.join(f'{results_path}', f'{hyperparameter}_evaluation.png')}.")