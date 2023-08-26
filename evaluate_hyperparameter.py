# Script to evaluate a certain pix2pix hyperparameter

import argparse
import os
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (e.g., results/hyperparameters/hyperparameter_netG)')
parser.add_argument('--ratio', type=float, default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')
parser.add_argument('--num_images', type=int, default=5, help='number of images to plot for evaluation')
parser.add_argument('--shuffle', action='store_true', help='if specified, shuffle the images before plotting')
parser.add_argument('--flip', action='store_true', help='if specified, make the rows correspond to different test images, not different models (i.e., rows and columns are flipped)')

results_path = parser.parse_args().results_path
ratio = parser.parse_args().ratio
num_images = parser.parse_args().num_images
shuffle = parser.parse_args().shuffle
flip = parser.parse_args().flip

hyperparameter = results_path.split('/')[-1].replace('hyperparameter_', '')

subfolders = [item for item in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, item))]

num_models = len(subfolders)

models = [item.replace(results_path.split('/')[-1] + '_', '') for item in subfolders]

# Part that needs to be inserted due to the way pix2pix saves the images
subpath_addition = 'test_latest/images'

images = [item for item in os.listdir(os.path.join(results_path, subfolders[0], subpath_addition)) if 'fake_B' in item]

if shuffle:
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

if flip:
    fig, ax = plt.subplots(limit, num_models+2, figsize=(width_per_image*(num_models+2), height_per_image*limit))

    ax[0, 0].text(0.5,1.1, 'fogged', transform=ax[0, 1].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k')
    ax[0, 1].text(0.5,1.1, 'original', transform=ax[0, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k')

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
                ax[i, j+2].text(0.5,1.1, models[j], transform=ax[i, j+2].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k')
            ax[i, j+2].axis('off')

else:
    fig, ax = plt.subplots(num_models+2, limit, figsize=(limit*width_per_image,(num_models+2)*height_per_image)) # num_models+2 to acommodate ground truth and fogged image

    ax[0, 0].text(-0.1,0.5, 'fogged', transform=ax[1, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k', rotation='vertical')
    ax[1, 0].text(-0.1,0.5, 'original', transform=ax[0, 0].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k', rotation='vertical')

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
                ax[j+2, i].text(-0.1,0.5, models[j], transform=ax[j+2, i].transAxes, backgroundcolor='w', horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='black', color='k', rotation='vertical')
            ax[j+2, i].axis('off')

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(os.path.join(f"{results_path}", f"{hyperparameter}_evaluation.png"), bbox_inches='tight', pad_inches=0)
    # img1 = plt.imread(results_path + images[i])
    # ax[3*i].imshow(img1, aspect='auto')
    # ax[3*i].axis('off')
    # plt.subplot(limit,3,2+3*i)
    # plt.title('real_A')
    
    # ax[1+3*i].imshow(img2, aspect='auto')

    # # Reading in the fogged image and calculating the variance of the Laplacian
    # fogged_image_gray = cv2.cvtColor(cv2.imread(results_path + images[i][:-10] + 'real_A' + '.png'), cv2.COLOR_BGR2GRAY)
    # fm = variance_of_laplacian(fogged_image_gray)

    # ax[1+3*i].text(0.5,0.03, f'Laplace: {fm:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor=cm.jet(norm(fm)), horizontalalignment='center', verticalalignment='bottom', fontweight='black', color='k' if fm > center_fog_value_limit else 'w')
    # ax[1+3*i].axis('off')
    # # plt.subplot(limit,3,3+3*i)
    # # plt.title('real_B')
    # img3 = plt.imread(results_path + images[i][:-10] + 'real_B' + '.png')
    # ax[2+3*i].imshow(img3, aspect='auto')
    # ax[2+3*i].axis('off')

    # # Reading in the clear image and calculating the SSIM to get a value for the fogginess (how much the fog changes the image) (https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images)
    # clear_image_gray = cv2.cvtColor(cv2.imread(results_path + images[i][:-10] + 'real_B' + '.png'), cv2.COLOR_BGR2GRAY)

    # (SSIM_score, SSIM_diff) = structural_similarity(img3, img2, full=True, multichannel=True)
    # ax[1+3*i].text(0.02,0.91, f'SSIM: {SSIM_score:.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='left', verticalalignment='bottom', fontweight='black', color='k')

    # # Calculating the Pearson correlation coefficient between the two images (https://stackoverflow.com/questions/34762661/percentage-difference-between-two-images-in-python-using-correlation-coefficient, https://mbrow20.github.io/mvbrow20.github.io/PearsonCorrelationPixelAnalysis.html)
    # Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
    # corrImAbs = np.absolute(Pearson_image_correlation)

    # ax[1+3*i].text(0.98,0.91, f'Pearson: {np.mean(corrImAbs):.2f}', transform=ax[1+3*i].transAxes, backgroundcolor='w', horizontalalignment='right', verticalalignment='bottom', fontweight='black', color='k')


# plt.figure(figsize=(15,10))

# plt.tight_layout()