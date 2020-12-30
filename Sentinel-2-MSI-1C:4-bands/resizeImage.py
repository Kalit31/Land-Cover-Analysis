import re
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from osgeo import gdal, osr

# To read the images in numerical order
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# Set file path accordingly
# List of file names of actual Satellite images for traininig
filelist_sat = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-sat/*.tif'), key=numericalSort)
# List of file names of classified images for traininig
filelist_trainy = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-gt/*.tif'), key=numericalSort)

print("Satellite SAR images: "+str(len(filelist_sat)))
print("Ground truths: "+str(len(filelist_trainy)))

for i, fname in enumerate(filelist_sat):
    tif = gdal.Open(filelist_sat[i])
    img = tif.ReadAsArray()

    image = Image.open(filelist_trainy[i])
    newImage = image.resize((img.shape[2], img.shape[1]))
    # Set file path accordingly
    newImage.save(
        '/home/data/IIRS/PS1-Project/src/GEE-Dataset-Processed/processed-resized-gt/' + str(i+1) + '.tif')
