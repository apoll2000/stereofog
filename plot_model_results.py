# Script for plotting the image results of the pix2pix model

from general_imports import *

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
from pytorch_msssim import ms_ssim
from PIL import Image
import torch

plt.rc('font', size=13)          # controls default text sizes

title_fontsize = 26
label_fontsize = 18

parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')
parser.add_argument('--num_images', type=int, default=5, help='number of images to plot')
parser.add_argument('--shuffle', action='store_true', help='if specified, shuffle the images before plotting')
parser.add_argument('--seed', type=int, default=16, help='seed for the random shuffling of the images')
parser.add_argument('--ratio', default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')
parser.add_argument('--no_laplace', action='store_true', help='if specified, do not plot the Laplacian variance on the fogged images')
parser.add_argument('--no_fog_colorbar', action='store_true', help='if specified, do not plot the fogginess colorbar below')
parser.add_argument('--sort_by_laplacian', action='store_true', help='if specified, sort the images by the Laplacian variance')
parser.add_argument('--specify_image', action='store_true', help='if specified, specify the image to plot')
parser.add_argument('--image_name', default='', help='name of the image to plot')
parser.add_argument('--model_type', type=str, default='pix2pix', help='type of model used (pix2pix or cycleGAN)')
parser.add_argument('--dataset_path', type=str, default='', help='path to the dataset for the cycleGAN evaluation')
parser.add_argument('--change_column_order', action='store_true', help='if specified, change the column order to foggy-predicted-clear instead of predicted-foggy-clear')
parser.add_argument('--dont_save_figure', action='store_true', help='if specified, only show figure, do not save it')
parser.add_argument('--add_title_indices', action='store_true', help='if specified, add title indices (like a) foggy, etc.')

args = parser.parse_args()

results_path = args.results_path
num_images = args.num_images
shuffle = args.shuffle
seed = args.seed
ratio = args.ratio
no_laplace = args.no_laplace
no_fog_colorbar = args.no_fog_colorbar
sort_by_laplacian = args.sort_by_laplacian
specify_image = args.specify_image
image_name = args.image_name
model_type = args.model_type
model_type = model_type.lower()
dataset_path = args.dataset_path
change_column_order = args.change_column_order
dont_save_figure = args.dont_save_figure
add_title_indices = args.add_title_indices

if model_type == 'pix2pix' and dataset_path != '':
    print('The dataset path is not used for the pix2pix model. Continuing...')
elif model_type == 'cyclegan' and dataset_path == '':
    raise ValueError('The dataset path must be specified for the cycleGAN model.')

if no_laplace:
    no_fog_colorbar = True

if type(ratio) != float:
    try:
        ratio = float(ratio.split('/')[0])/float(ratio.split('/')[1])
    except:
        raise ValueError('The ratio must be a float or a string of the form "x/y".')

original_results_path = results_path
results_path = os.path.join(results_path, 'test_latest/images')

# CW-SSIM implementation
gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

# Indexing the images
if specify_image:
    images = [image_name]
    num_images = 1
else:
    if model_type == 'pix2pix':
        images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

    elif model_type == 'cyclegan':
        images = [entry for entry in os.listdir(results_path) if 'fake' in entry]

    else:
        raise ValueError('The model type must be either "pix2pix" or "cycleGAN".')

# Defining commonly used variables
if model_type == 'pix2pix':
    real_foggy_image_addition = 'real_A'
    real_clear_image_addition = 'real_B'
    letters_to_remove = 10
elif model_type == 'cyclegan':
    real_foggy_image_addition = 'real'
    dataset_path = os.path.join(dataset_path, 'testA')
    letters_to_remove = 8


# Shuffling the images if specified
if shuffle:
    random.seed(seed)
    random.shuffle(images)

# Setting width and height
width_per_image = 4
height_per_image = width_per_image / ratio

images = images[:num_images]

# Computing the Variance of the Laplacian for each of the images
if not no_laplace:

    laplacian_values = []
    for i in range(num_images):
        # Reading in the fogged image and calculating the variance of the Laplacian
        fogged_image_gray = cv2.cvtColor(cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png')), cv2.COLOR_BGR2GRAY)
        
        laplacian_values += [variance_of_laplacian(fogged_image_gray)]

# Sorting the images by the Laplacian variance if specified
if sort_by_laplacian:
    both = sorted(zip(laplacian_values, images))
    laplacian_values = [laplacian_value for laplacian_value, image in both]
    images = [image for laplacian_value, image in both]

if not no_fog_colorbar or not no_laplace:
    # Creating the colormap for the fogginess of the images
    min_fog_value_limit = min(laplacian_values)
    max_fog_value_limit = max(laplacian_values)
    center_fog_value_limit = min_fog_value_limit + (max_fog_value_limit - min_fog_value_limit)*0.4
    norm = matplotlib.colors.Normalize(vmin=min_fog_value_limit, vmax=max_fog_value_limit)    # Normalizer for the values of the colormap rating the fogginess of the image: https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib

if not no_fog_colorbar:
    # Creating the overarching figure containing the images and the colormap (from here: https://stackoverflow.com/questions/34933905/adding-subplots-to-a-subplot)
    superfig = plt.figure(figsize=(3*width_per_image,((num_images + 0.1)*height_per_image)))

    # Creating the subfigures for the images and the colormap
    subfigs = superfig.subfigures(2, 1, height_ratios=[num_images, 0.1*num_images])

    # Creating the subplots for the images
    ax = [subfigs[0].add_subplot(num_images,3,i+1) for i in range(num_images*3)]

else:
    # Creating the overarching figure containing the images and the colormap (from here: https://stackoverflow.com/questions/34933905/adding-subplots-to-a-subplot)
    superfig = plt.figure(figsize=(3*width_per_image,((num_images)*height_per_image)))

    # Creating the subfigures for the images and the colormap
    subfigs = superfig.subfigures(1, 1)#, height_ratios=[num_images, 0.1*num_images])

    # Creating the subplots for the images
    ax = [subfigs.add_subplot(num_images,3,i+1) for i in range(num_images*3)]

# Setting the titles for the images
if add_title_indices:
    title_pre_string_a = 'a) '
    title_pre_string_b = 'b) '
    title_pre_string_c = 'c) '
else:
    title_pre_string_a = title_pre_string_b = title_pre_string_c = ''

if not change_column_order:
    ax[0].text(0.5, 1.05, f'{title_pre_string_a}reconstructed', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[0].transAxes)
    ax[1].text(0.5, 1.05, f'{title_pre_string_b}foggy real', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[1].transAxes)
    ax[2].text(0.5, 1.05, f'{title_pre_string_c}ground truth', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[2].transAxes)
else:
    ax[0].text(0.5, 1.05, f'{title_pre_string_a}foggy real', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[0].transAxes)
    ax[1].text(0.5, 1.05, f'{title_pre_string_b}reconstructed', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[1].transAxes)
    ax[2].text(0.5, 1.05, f'{title_pre_string_c}ground truth', fontsize=title_fontsize, color='k', fontweight='black', ha='center', transform=ax[2].transAxes)

for i in range(num_images):
    # Reading in the fake image
    img1 = plt.imread(os.path.join(results_path, images[i]))
    if not change_column_order:
        ax[3*i].imshow(img1, aspect='auto')
        ax[3*i].axis('off')
    else:
        ax[1+3*i].imshow(img1, aspect='auto')
        ax[1+3*i].axis('off')

    # Reading in the fogged image
    if not change_column_order:
        img2 = plt.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png'))
        ax[1+3*i].imshow(img2, aspect='auto')
    else:
        img2 = plt.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png'))
        ax[3*i].imshow(img2, aspect='auto')

    # Reading in the fogged image again and calculating the variance of the Laplacian
    fogged_image_gray = cv2.cvtColor(cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png')), cv2.COLOR_BGR2GRAY)
    
    if not no_laplace:
        # fm = variance_of_laplacian(fogged_image_gray)
        # Putting the value of the variance of the Laplacian on the image
        if not change_column_order:
            ax[1+3*i].text(0.5,0.03, '$\mathbf{v_{L}}$: %.2f' %laplacian_values[i], transform=ax[1+3*i].transAxes, backgroundcolor=cm.jet_r(norm(laplacian_values[i])), horizontalalignment='center', verticalalignment='bottom', fontsize=label_fontsize, fontweight='black', color='k' if (laplacian_values[i] > center_fog_value_limit and laplacian_values[i] < center_fog_value_limit+(max_fog_value_limit - min_fog_value_limit)*0.5) else 'w')
        else:
            ax[3*i].text(0.5,0.03, '$\mathbf{v_{L}}$: %.2f' %laplacian_values[i], transform=ax[3*i].transAxes, backgroundcolor=cm.jet_r(norm(laplacian_values[i])), horizontalalignment='center', verticalalignment='bottom', fontsize=label_fontsize, fontweight='black', color='k' if (laplacian_values[i] > center_fog_value_limit and laplacian_values[i] < center_fog_value_limit+(max_fog_value_limit - min_fog_value_limit)*0.5) else 'w')

    if not change_column_order:
        ax[1+3*i].axis('off')
    else:
        ax[3*i].axis('off')

    # Reading in the clear image
    if model_type == 'pix2pix':
        img3 = plt.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))
    elif model_type == 'cyclegan':
        img3 = plt.imread(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png'))
    ax[2+3*i].imshow(img3, aspect='auto')
    ax[2+3*i].axis('off')

    # Reading in the clear image again and calculating the SSIM to get a value for the fogginess (how much the fog changes the image) (https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images)
    fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png'))
    if model_type == 'pix2pix':
        clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))
    elif model_type == 'cyclegan':
        h, w, c = fogged_image_nonfloat.shape
        clear_image_nonfloat = cv2.resize(cv2.imread(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png')), (w, h))

    fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

    (SSIM_score, SSIM_diff) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)
    # Putting the value of the SSIM on the image
    if not change_column_order:
        ax[1+3*i].text(0,1, f'SSIM: {SSIM_score:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)
    else:
        ax[3*i].text(0,1, f'SSIM: {SSIM_score:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)

    # Calculating the Pearson correlation coefficient between the two images (https://stackoverflow.com/questions/34762661/percentage-difference-between-two-images-in-python-using-correlation-coefficient, https://mbrow20.github.io/mvbrow20.github.io/PearsonCorrelationPixelAnalysis.html)
    clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
    Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
    corrImAbs = np.absolute(Pearson_image_correlation)
    # Putting the value of the Pearson correlation coefficient on the image
    if not change_column_order:
        ax[1+3*i].text(1,1, f'Pearson: {np.mean(corrImAbs):.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)
    else:
        ax[3*i].text(1,1, f'Pearson: {np.mean(corrImAbs):.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)


    # Calculating the SSIM between the fake image and the clear image
    (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fake_image_nonfloat, full=True, multichannel=True, channel_axis=2)
    # Putting the value of the SSIM on the fake image
    if not change_column_order:
        ax[3*i].text(0,1, f'SSIM (r): {SSIM_score_reconstruction:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)
    else:
        ax[1+3*i].text(0,1, f'SSIM (r): {SSIM_score_reconstruction:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)

    # Calculating the CW-SSIM between the fake image and the clear image (https://github.com/jterrace/pyssim)
    if model_type == 'pix2pix':
        CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))
    elif model_type == 'cyclegan':
        CW_SSIM = 0#SSIM(Image.open(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))
    # Putting the value of the CW-SSIM on the fake image
    if not change_column_order:
        ax[3*i].text(1,1, f'CW-SSIM (r): {CW_SSIM:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)
    else:
        ax[1+3*i].text(1,1, f'CW-SSIM (r): {CW_SSIM:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='top', fontweight='black', color='k', fontsize=label_fontsize)

    # Calculating the MS-SSIM between the fake image and the clear image
    if model_type == 'pix2pix':
        real_img = np.array(Image.open(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))).astype(np.float32)
    elif model_type == 'cyclegan':
        real_img = np.array(Image.open(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png')).resize((w, h))).astype(np.float32)
    real_img = torch.from_numpy(real_img).unsqueeze(0).permute(0, 3, 1, 2)
    fake_img = np.array(Image.open(os.path.join(results_path, images[i]))).astype(np.float32)
    fake_img = torch.from_numpy(fake_img).unsqueeze(0).permute(0, 3, 1, 2)
    MS_SSIM = ms_ssim(real_img, fake_img, data_range=255, size_average=True).item()
    # Putting the value of the MS-SSIM on the fake image
    if not change_column_order:
        ax[3*i].text(1,0, f'MS-SSIM (r): {MS_SSIM:.2f}', transform=ax[3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='bottom', fontweight='black', color='k', fontsize=label_fontsize)
    else:
        ax[1+3*i].text(1,0, f'MS-SSIM (r): {MS_SSIM:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='bottom', fontweight='black', color='k', fontsize=label_fontsize)

# plt.figure(figsize=(15,10))

if not no_fog_colorbar:
    # Plotting the colormap below (https://stackoverflow.com/questions/2451264/creating-a-colormap-legend-in-matplotlib)
    m = np.zeros((1,200))
    for i in range(200):
        m[0,i] = (i)/200.0

    plt.subplots_adjust(hspace=0, wspace=0)

    ax = subfigs[1].add_subplot(1, 1, 1)

    plt.imshow(m, cmap='jet_r', aspect = 2)
    plt.yticks(np.arange(0))

    ax.axis('off')

    # Adding labels to the colormap
    for coordinate, text, ha in zip([0, 100, 200], [f'high fog ({min_fog_value_limit:.2f})', f'medium fog ({min_fog_value_limit + (max_fog_value_limit - min_fog_value_limit)*0.4:.2f})', f'low fog ({max_fog_value_limit:.2f})'], ['left', 'center', 'right']):
        plt.text(coordinate, -0.7, text, ha=ha, va='bottom', fontweight='black', color='k', fontsize=title_fontsize)

    superfig.tight_layout()

else:
    # plt.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout()

if not dont_save_figure:
    if specify_image:
        plt.savefig(os.path.join(original_results_path, f"{original_results_path.split('/')[-1]}_evaluation_{image_name}.pdf"), format='pdf', bbox_inches='tight')
        print("Saved the evaluation plot to", os.path.join(original_results_path, f"{original_results_path.split('/')[-1]}_evaluation_{image_name}.pdf."))
    else:
        plt.savefig(os.path.join(original_results_path, f"{original_results_path.split('/')[-1]}_evaluation.pdf"), format='pdf', bbox_inches='tight')
        print("Saved the evaluation plot to", os.path.join(original_results_path, f"{original_results_path.split('/')[-1]}_evaluation.pdf."))
else:
    plt.show()