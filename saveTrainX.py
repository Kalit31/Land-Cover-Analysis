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
filelist_trainx = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-sat/*.tif'), key=numericalSort)

# Set of datasets collected.
# 0000-5888
# 0000-0000
# 5888-5888-1
# 5888-11776-1


# Reading, padding, cropping and making array of all the cropped images of all the trainig sat images
trainx_cropped = getCroppedImages(filelist_trainx)
np.save('/home/data/IIRS/PS1-Project/src/trainX.npy', trainx_cropped)
print('Saved cropped training X')
