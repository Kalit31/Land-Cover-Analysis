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


# List of actual Ground Truth Labels for training
# Set the input data path appropriately
filelist_trainy = sorted(glob.glob(
    '/home/data/IIRS/December-Testing/Dataset/processed-resized-gt/*.tif'), key=numericalSort)

print(str(len(filelist_trainy))+" images found.")

filelist_trainy_filtered = filelist_trainy[0:21]

print("Saving "+str(len(filelist_trainy_filtered))+" images.")

# Reading, padding, cropping and making array of all the cropped images of all the training gt images
trainy_cropped = getTrainingCroppedImagesMorethan1Band(
    filelist_trainy_filtered)

# Convert the RGB data to One hot encoded data
trainy_hot = convertRGBtoOneHot(trainy_cropped)

# Save data in a numpy array file to be used later
# Specify path appropriately
np.save('/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/NumpyArrays/trainYHotVal-sar.npy', trainy_hot)

print('Saved training hot Y')
