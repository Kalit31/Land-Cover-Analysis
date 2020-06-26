import re
import numpy as np
import glob
import math
from image_processing_functions import *

# To read the images in numerical order
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# List of actual Ground Truth Labels for training
filelist_trainy = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-resized-gt-4layer/*.tif'), key=numericalSort)

# Reading, padding, cropping and making array of all the cropped images of all the trainig gt images
trainy_cropped = getCroppedImages(filelist_trainy)
trainy_hot = convertRGBtoOneHot(trainy_cropped)
np.save('/home/data/IIRS/PS1-Project/src/trainYHot.npy', trainy_hot)
print('Saved cropped training HOT Y')
