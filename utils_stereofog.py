# -*- coding: utf-8 -*-
"""
Created on Tue 15 Aug 2023

Utility functions by Anton Pollak used in the stereofog project


"""

import cv2

# code for detecting the blurriness of an image (https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return -cv2.Laplacian(image, cv2.CV_32F).var()