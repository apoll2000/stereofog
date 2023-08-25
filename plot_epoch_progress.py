import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import gif


parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--model_name', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--verbosity', type=int, default=1, help='verbosity level (i.e., print every `verbosity` epochs)')
parser.add_argument('--image_index', type=int, default=0, help='try another image if the first one is not suitable')
parser.add_argument('--output_type', type=str, default='image', help='output type (image or gif)')
args = parser.parse_args()

dataroot = args.dataroot
model_name = args.model_name
verbosity = args.verbosity
image_index = args.image_index
output_type = args.output_type

# Function for producing each of the frame of the gif (documentation: https://pypi.org/project/gif/)
@gif.frame
def produce_fig(epoch_index, image_index):

    epoch = epochs[::verbosity][epoch_index]

    subpath = os.path.join(results_path, f'{model_name}_epochs{epoch}', model_name, f'test_{epoch}', 'images')

    try:
        file = [item for item in os.listdir(subpath) if 'fake' in item][image_index]
    except:
        print(f'Image index {image_index} not found. Using default image index 0 instead.')
        file = [item for item in os.listdir(subpath) if 'fake' in item][0]
        
        image_index = 0 # make sure that all images use this index now

    img = plt.imread(os.path.join(subpath, file))


    fig, ax = plt.subplots(1, 3, figsize=(3*width_per_image, height_per_image))

    ax[0].imshow(img, aspect='auto')
    ax[0].axis('off')
    ax[0].text(0., 0., f'epoch {epoch}', color='k', backgroundcolor='w', horizontalalignment='left', verticalalignment='bottom', transform=ax[0].transAxes)

    ax[1].imshow(original_img_A, aspect='auto')
    ax[1].axis('off')

    ax[2].imshow(original_img_B, aspect='auto')
    ax[2].axis('off')

    plt.subplots_adjust(hspace=0, wspace=0)

    return fig

# model_name = 'stereofog_pix2pix'
# dataroot = 'datasets/stereofog_images'

checkpoints_directory = f'checkpoints/{model_name}'




epochs = sorted(list(set([int(item.replace('_net_G.pth', '').replace('_net_D.pth', '')) for item in os.listdir(checkpoints_directory) if '.pth' in item and 'latest' not in item])))

results_path = f'results/epochs/'

dataset_image_path = os.path.join(dataroot, 'A', 'test')
dataset_image_file = [item for item in os.listdir(dataset_image_path)][0]
dataset_image = plt.imread(os.path.join(dataset_image_path, dataset_image_file))

ratio = dataset_image.shape[1] / dataset_image.shape[0]

width_per_image = 4
height_per_image = width_per_image / ratio

original_subpath_A = os.path.join(results_path, f'{model_name}_epochs{epochs[0]}', model_name, f'test_{epochs[0]}', 'images')
original_file_A = [item for item in os.listdir(original_subpath_A) if 'real_B' in item][image_index]
original_img_A = plt.imread(os.path.join(original_subpath_A, original_file_A))


original_subpath_B = os.path.join(results_path, f'{model_name}_epochs{epochs[0]}', model_name, f'test_{epochs[0]}', 'images')
original_file_B = [item for item in os.listdir(original_subpath_B) if 'real_A' in item][image_index]
original_img_B = plt.imread(os.path.join(original_subpath_B, original_file_B))


# fig, ax = plt.subplots(len(epochs), 3, figsize=(3*width_per_image, len(epochs)*height_per_image))
# ax = ax.flatten()

# checking if the save directory exists
if not os.path.exists(f"results/epochs/{model_name}_epochs_results"):
    os.makedirs(f"results/epochs/{model_name}_epochs_results")

if output_type == 'image':

    fig = plt.figure(figsize=(3*width_per_image,len(epochs[::verbosity])*height_per_image))

    ax = [fig.add_subplot(len(epochs[::verbosity]),3,i+1) for i in range(len(epochs[::verbosity])*3)]

    for i, epoch in enumerate(epochs[::verbosity]):

        subpath = os.path.join(results_path, f'{model_name}_epochs{epoch}', model_name, f'test_{epoch}', 'images')

        try:
            file = [item for item in os.listdir(subpath) if 'fake' in item][image_index]
        except:
            print(f'Image index {image_index} not found. Using default image index 0 instead.')
            file = [item for item in os.listdir(subpath) if 'fake' in item][0]
            
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
    plt.savefig(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.png")
    print(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.png saved successfully.")


elif output_type == 'gif':
    # fastgif.make_gif(produce_fig, len(epochs[::verbosity]), 'test.gif', show_progress=True, writer_kwargs={'image_index': image_index})

    frames = [produce_fig(i, image_index) for i in range(len(epochs[::verbosity]))]

    gif.save(frames, f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.gif", duration=300)
    print(f"results/epochs/{model_name}_epochs_results/{model_name}_epochs_results.gif saved successfully.")