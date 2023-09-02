# Script for plotting the image results of the pix2pix model

import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from utils_stereofog import variance_of_laplacian
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')
parser.add_argument('--num_images', type=int, default=5, help='number of images to plot')
parser.add_argument('--shuffle', action='store_true', help='if specified, shuffle the images before plotting')
parser.add_argument('--seed', type=int, default=16, help='seed for the random shuffling of the images')
parser.add_argument('--ratio', type=float, default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')

args = parser.parse_args()

results_path = args.results_path
num_images = args.num_images
shuffle = args.shuffle
seed = args.seed
ratio = args.ratio

original_results_path = results_path
results_path = os.path.join(results_path, 'test_latest/images')

# Creating the colormap for the fogginess of the images
min_fog_value_limit = -7
max_fog_value_limit = -5
center_fog_value_limit = min_fog_value_limit + (max_fog_value_limit - min_fog_value_limit)*0.4
norm = matplotlib.colors.Normalize(vmin=min_fog_value_limit, vmax=max_fog_value_limit)    # Normalizer for the values of the colormap rating the fogginess of the image: https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib

# CW-SSIM implementation
gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

# Indexing the images
images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

# Shuffling the images if specified
if shuffle:
    random.seed(seed)
    random.shuffle(images)

# Setting width and height
width_per_image = 4
height_per_image = width_per_image / ratio

# Creating the overarching figure containing the images and the colormap (from here: https://stackoverflow.com/questions/34933905/adding-subplots-to-a-subplot)
superfig = plt.figure(figsize=(3*width_per_image,((num_images + 0.1)*height_per_image)))

# Creating the subfigures for the images and the colormap
subfigs = superfig.subfigures(2, 1, height_ratios=[num_images, 0.1*num_images])

# Creating the subplots for the images
ax = [subfigs[0].add_subplot(num_images,3,i+1) for i in range(num_images*3)]

# Setting the titles for the images
ax[0].text(0.5, 1.05, 'fake', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[0].transAxes)
ax[1].text(0.5, 1.05, 'foggy real', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[1].transAxes)
ax[2].text(0.5, 1.05, 'clear real', fontsize=15, color='k', fontweight='black', ha='center', transform=ax[2].transAxes)

for i in range(num_images):
    # Reading in the fake image
    img1 = plt.imread(os.path.join(results_path, images[i]))
    ax[3*i].imshow(img1, aspect='auto')
    ax[3*i].axis('off')

    # Reading in the fogged image
    img2 = plt.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
    ax[1+3*i].imshow(img2, aspect='auto')

    # Reading in the fogged image again and calculating the variance of the Laplacian
    fogged_image_gray = cv2.cvtColor(cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png')), cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(fogged_image_gray)
    # Putting the value of the variance of the Laplacian on the image
    ax[1+3*i].text(0.5,0.03, f'Laplace: {fm:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor=cm.jet(norm(fm)), horizontalalignment='center', verticalalignment='bottom', fontweight='black', color='k' if fm > center_fog_value_limit else 'w')
    ax[1+3*i].axis('off')

    # Reading in the clear image
    img3 = plt.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
    ax[2+3*i].imshow(img3, aspect='auto')
    ax[2+3*i].axis('off')

    # Reading in the clear image again and calculating the SSIM to get a value for the fogginess (how much the fog changes the image) (https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images)
    clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
    fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
    fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

    (SSIM_score, SSIM_diff) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)
    # Putting the value of the SSIM on the image
    ax[1+3*i].text(0,1, f'SSIM: {SSIM_score:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k')
    
    # Calculating the Pearson correlation coefficient between the two images (https://stackoverflow.com/questions/34762661/percentage-difference-between-two-images-in-python-using-correlation-coefficient, https://mbrow20.github.io/mvbrow20.github.io/PearsonCorrelationPixelAnalysis.html)
    clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
    Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
    corrImAbs = np.absolute(Pearson_image_correlation)
    # Putting the value of the Pearson correlation coefficient on the image
    ax[1+3*i].text(1,1, f'Pearson: {np.mean(corrImAbs):.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k')

    # Calculating the SSIM between the fake image and the clear image
    (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)
    # Putting the value of the SSIM on the fake image
    ax[3*i].text(0,1, f'SSIM (r): {SSIM_score_reconstruction:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k')

    # Calculating the CW-SSIM between the fake image and the clear image (https://github.com/jterrace/pyssim)
    CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))
    # Putting the value of the CW-SSIM on the fake image
    ax[3*i].text(1,1, f'CW-SSIM (r): {CW_SSIM:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k')

# plt.figure(figsize=(15,10))


# Plotting the colormap below (https://stackoverflow.com/questions/2451264/creating-a-colormap-legend-in-matplotlib)
m = np.zeros((1,200))
for i in range(200):
    m[0,i] = (i)/200.0

ax = subfigs[1].add_subplot(1, 1, 1)

plt.imshow(m, cmap='jet', aspect = 2)
plt.yticks(np.arange(0))

ax.axis('off')

# Adding labels to the colormap
for coordinate, text, ha in zip([0, 100, 200], ['low fog', 'medium fog', 'high fog'], ['left', 'center', 'right']):
    plt.text(coordinate, -0.7, text, ha=ha, va='bottom', fontweight='black', color='k')

superfig.tight_layout()

plt.savefig(os.path.join(original_results_path, f"{original_results_path.split('/')[-1]}_evaluation.png"), bbox_inches='tight')