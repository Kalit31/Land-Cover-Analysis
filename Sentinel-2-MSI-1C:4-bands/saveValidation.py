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


# Set the input data path appropriately
# List of actual Satellite images for traininig
filelist_trainx = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-sat-valid/*.tif'), key=numericalSort)
# List of actual Ground Truth Labels for training
filelist_trainy = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-resized-gt-4layer-valid/*.tif'), key=numericalSort)


filelist_trainx_filtered = filelist_trainx[21:]
train_val_cropped = getTrainingCroppedImagesMorethan1Band(
    filelist_trainx_filtered)
np.save('/home/data/IIRS/December-Testing/NumpyArrays/trainXVal-sar.npy',
        train_val_cropped)
print('Saved validation X')

filelist_trainy_filtered = filelist_trainy[21:]
trainy_val_cropped = getTrainingCroppedImagesMorethan1Band(
    filelist_trainy_filtered)
trainy_hot = convertRGBtoOneHot(trainy_val_cropped)
np.save('/home/data/IIRS/December-Testing/NumpyArrays/trainYValHot.npy', trainy_hot)
print('Saved validation hot Y')
