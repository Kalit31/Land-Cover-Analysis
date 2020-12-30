import re
from osgeo import gdal, osr
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import math
from image_processing_functions import *

# To read the images in numerical order
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# List of actual Satellite images for traininig
# Set the input data path appropriately
filelist_trainx = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-sat/*.tif'), key=numericalSort)


print(str(len(filelist_trainx))+" images found.")

filelist_trainx_filtered = filelist_trainx[0:21]

print("Saving "+str(len(filelist_trainx_filtered))+" images.")

# Reading, padding, cropping and making array of all the cropped images of all the training sat images
trainx_cropped = getTrainingCroppedImagesMorethan1Band(
    filelist_trainx_filtered)
np.save('/home/data/IIRS/PS1-Project/src/trainX.npy', trainx_cropped)

print("Saved Training X")
