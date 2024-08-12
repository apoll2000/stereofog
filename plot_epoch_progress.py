import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import gif
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image
from utils_stereofog import calculate_model_results, generate_stats_from_log
from general_imports import *

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints_path', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--model_name', type=str, default=None, help='Override model name (if it should not be inferred from the dataroot))')
parser.add_argument('--verbosity', type=int, default=5, help='verbosity level (i.e., print every `verbosity` epochs)')
parser.add_argument('--ratio', type=float, default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')
parser.add_argument('--image_index', type=int, default=0, help='try another image if the first one is not suitable')
parser.add_argument('--output_type', type=str, default='image', help='output type (image or gif)')
args = parser.parse_args()

# dataroot = args.dataroot
checkpoints_path = args.checkpoints_path
if args.model_name is None:
    model_name = checkpoints_path.split('/')[-1]
else:
    model_name = args.model_name
verbosity = args.verbosity
ratio = args.ratio
image_index = args.image_index
output_type = args.output_type

# Function for producing each of the frame of the gif (documentation: https://pypi.org/project/gif/)
@gif.frame
def produce_fig(epoch_index, image_index):

    epoch = epochs[::verbosity][epoch_index]

    subpath = os.path.join(results_path, f'{epoch}', model_name, f'test_{epoch}', 'images')

    try:
        file = sorted([item for item in os.listdir(subpath) if 'fake' in item])[image_index]
    except:
        print(f'Image index {image_index} not found. Using default image index 0 instead.')
        file = sorted([item for item in os.listdir(subpath) if 'fake' in item])[0]
        
        image_index = 0 # make sure that all images use this index now

    img = plt.imread(os.path.join(subpath, file))


    fig, ax = plt.subplots(1, 5, figsize=(5*width_per_image, height_per_image))

    ax[0].plot(epochs[::verbosity], mean_CW_SSIM_scores)
    ax[0].scatter(epochs[::verbosity][epoch_index], mean_CW_SSIM_scores[epoch_index], color='r')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('CW-SSIM score')

    ax[1].imshow(img, aspect='auto')
    ax[1].axis('off')
    ax[1].text(0., 0., f'epoch {epoch}', color='k', backgroundcolor='w', horizontalalignment='left', verticalalignment='bottom', transform=ax[1].transAxes)
    ax[1].text(0.5, 1, 'fake', color='k', backgroundcolor='w', horizontalalignment='center', verticalalignment='top', transform=ax[1].transAxes)

    ax[2].imshow(original_img_A, aspect='auto')
    ax[2].axis('off')
    ax[2].text(0.5, 1, 'clear real', color='k', backgroundcolor='w', horizontalalignment='center', verticalalignment='top', transform=ax[2].transAxes)

    ax[3].imshow(original_img_B, aspect='auto')
    ax[3].axis('off')
    ax[3].text(0.5, 1, 'foggy real', color='k', backgroundcolor='w', horizontalalignment='center', verticalalignment='top', transform=ax[3].transAxes)

    _, ax[4] = generate_stats_from_log(checkpoints_path, fig=fig, ax=ax[4], highlight_epoch=epoch)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout()

    return fig

# model_name = 'stereofog_pix2pix'
# dataroot = 'datasets/stereofog_images'

# checkpoints_directory = f'checkpoints/{model_name}'
# checkpoints_directory = dataroot.replace('datasets', 'checkpoints')

# Extracting the epochs from the checkpoints directory
epochs = sorted(list(set([int(item.replace('_net_G.pth', '').replace('_net_D.pth', '')) for item in os.listdir(checkpoints_path) if '.pth' in item and 'latest' not in item])))

# results_path = f'results/epochs/'
results_path = f"results/epochs/{model_name}_epochs"

# dataset_image_path = os.path.join(dataroot, 'A', 'test')
# dataset_image_file = [item for item in os.listdir(dataset_image_path)][0]
# dataset_image = plt.imread(os.path.join(dataset_image_path, dataset_image_file))

# ratio = dataset_image.shape[1] / dataset_image.shape[0]

# Setting width and height for the plots
width_per_image = 4
height_per_image = width_per_image / ratio

# Reading in the original images (these will just be replotted again for every epoch and do not change)
original_subpath_A = os.path.join(results_path, f'{epochs[0]}', model_name, f'test_{epochs[0]}', 'images')
original_file_A = sorted([item for item in os.listdir(original_subpath_A) if 'real_B' in item])[image_index]
original_img_A = plt.imread(os.path.join(original_subpath_A, original_file_A))

original_subpath_B = os.path.join(results_path, f'{epochs[0]}', model_name, f'test_{epochs[0]}', 'images')
original_file_B = sorted([item for item in os.listdir(original_subpath_B) if 'real_A' in item])[image_index]
original_img_B = plt.imread(os.path.join(original_subpath_B, original_file_B))


# fig, ax = plt.subplots(len(epochs), 3, figsize=(3*width_per_image, len(epochs)*height_per_image))
# ax = ax.flatten()

# Evaluating the CW-SSIM score for each epoch
gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

mean_CW_SSIM_scores = []

for i, epoch in enumerate(epochs[::verbosity]):    
    # subpath = os.path.join(results_path, f'{epoch}', model_name, f'test_{epoch}', 'images')

    # try:
    #     file = [item for item in os.listdir(subpath) if 'fake' in item][image_index]
    # except:
    #     print(f'Image index {image_index} not found. Using default image index 0 instead.')
    #     file = [item for item in os.listdir(subpath) if 'fake' in item][0]
        
    #     image_index = 0 # make sure that all images use this index now

    mean_Pearson, mean_MSE, mean_PSNR, mean_NCC, mean_SSIM, mean_CW_SSIM, mean_MS_SSIM = calculate_model_results(os.path.join(results_path, str(epoch)), epoch=epoch, epoch_test=True)

    mean_CW_SSIM_scores.append(mean_CW_SSIM)




# checking if the save directory exists
if not os.path.exists(f"results/epochs/{model_name}_epochs_results"):
    os.makedirs(f"results/epochs/{model_name}_epochs_results")

if output_type == 'image':

    fig = plt.figure(figsize=(3*width_per_image,len(epochs[::verbosity])*height_per_image))

    ax = [fig.add_subplot(len(epochs[::verbosity]),3,i+1) for i in range(len(epochs[::verbosity])*3)]

    for i, epoch in enumerate(epochs[::verbosity]):

        subpath = os.path.join(results_path, f'{epoch}', model_name, f'test_{epoch}', 'images')

        try:
            file = sorted([item for item in os.listdir(subpath) if 'fake' in item])[image_index]
        except:
            print(f'Image index {image_index} not found. Using default image index 0 instead.')
            file = sorted([item for item in os.listdir(subpath) if 'fake' in item])[0]
            
            image_index = 0 # make sure that all images use this index now

        img = plt.imread(os.path.join(subpath, file))

        ax[3*i].imshow(img, aspect='auto')
        ax[3*i].axis('off')
        ax[3*i].text(0.1, 0.02, f'epoch {epoch}', color='k', backgroundcolor='w', horizontalalignment='center', verticalalignment='center', transform=ax[3*i].transAxes)

        ax[3*i+2].imshow(original_img_A, aspect='auto')
        ax[3*i+2].axis('off')

        ax[3*i+1].imshow(original_img_B, aspect='auto')
        ax[3*i+1].axis('off')

    plt.subplots_adjust(hspace=0, wspace=0)
    # plt.savefig(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.png")
    plt.show()
    print(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.png saved successfully.")

    plt.plot(epochs[::verbosity], mean_CW_SSIM_scores)
    # plt.savefig(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results_scores.png")
    plt.show()


elif output_type == 'gif':
    # fastgif.make_gif(produce_fig, len(epochs[::verbosity]), 'test.gif', show_progress=True, writer_kwargs={'image_index': image_index})

    frames = [produce_fig(i, image_index) for i in range(len(epochs[::verbosity]))]

    gif.save(frames, f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.gif", duration=500)
    print(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.gif saved successfully.")