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


# List of actual Satellite images for traininig
filelist_validx = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-sat-valid/*.tif'), key=numericalSort)
# List of actual Ground Truth Labels for training
filelist_validy = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-resized-gt-4layer-valid/*.tif'), key=numericalSort)

train_val_cropped = getCroppedImages(filelist_validx)
np.save('/home/data/IIRS/PS1-Project/src/trainXVal.npy', train_val_cropped)
print('Saved cropped validation X')


trainy_val_cropped = getCroppedImages(filelist_validy)
trainy_hot = convertRGBtoOneHot(trainy_val_cropped)
np.save('/home/data/IIRS/PS1-Project/src/trainYValHot.npy', trainy_hot)
print('Saved cropped validation HOT Y')
