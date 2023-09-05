# Python file to evaluate the SSIMs of a model results folder

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image
import cv2
from utils_stereofog import calculate_model_results


parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')

args = parser.parse_args()

results_path = args.results_path


# def calculate_model_results(results_path = results_path):

#     results_path = os.path.join(results_path, 'test_latest/images')

#     # CW-SSIM implementation
#     gaussian_kernel_sigma = 1.5
#     gaussian_kernel_width = 11
#     gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

#     # Indexing the images
#     images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

#     SSIM_scores = []
#     CW_SSIM_scores = []
#     Pearson_image_correlations = []

#     for i, image in enumerate(images):


#         clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
#         fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
#         fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

#         # Calculating the Pearson correlation coefficient between the two images (https://stackoverflow.com/questions/34762661/percentage-difference-between-two-images-in-python-using-correlation-coefficient, https://mbrow20.github.io/mvbrow20.github.io/PearsonCorrelationPixelAnalysis.html)
#         # clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
#         # Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
#         # corrImAbs = np.absolute(Pearson_image_correlation)

#         # Pearson_image_correlations.append(np.mean(corrImAbs))

#         # Calculating the SSIM between the fake image and the clear image
#         (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)

#         SSIM_scores.append(SSIM_score_reconstruction)

#         # Calculating the CW-SSIM between the fake image and the clear image (https://github.com/jterrace/pyssim)
#         CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))

#         CW_SSIM_scores.append(CW_SSIM)

#         # Calculate the average values

#         mean_SSIM = np.mean(SSIM_scores)
#         mean_CW_SSIM = np.mean(CW_SSIM_scores)

#         return mean_SSIM, mean_CW_SSIM

mean_SSIM, mean_CW_SSIM = calculate_model_results(results_path)

print(f"Mean SSIM: {mean_SSIM:.2f}")
print(f"Mean CW-SSIM: {mean_CW_SSIM:.2f}")