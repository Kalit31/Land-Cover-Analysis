import re
from osgeo import gdal, osr
import PIL
from PIL import Image
import numpy as np
import glob
from image_processing_functions import *
from RGBOneHotConv import *
from unet_model import UNet_4Band

# Load model
model = UNet_4Band()

weights_file = "/home/data/IIRS/PS1-Project/src/trained_model.h5"

# Load trained weights
model.load_weights(weights_file)

# To read the images in numerical order
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# List of actual Satellite images for traininig
filelist = sorted(glob.glob(
    '/home/data/IIRS/PS1-Project/src/Jun03-04/processed-sat/*.tif'), key=numericalSort)

for k, fname in enumerate(filelist):

    # Reading the image
    image = readTiffImage4Band(fname)

    # Process image
    crop_size = 128
    stride = 32
    h, w, c = image.shape
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    image = padding(image, w, h, crop_size, stride, n_h, n_w, c)

    h, w, c = image.shape

    # Form an array of one image
    # This is done because, the model takes in an array of images as input.
    item = np.reshape(image, (1, h, w, c))

    # Predict the segmentation for the image
    y_pred_img = model.predict(item)

    # Predicted image shape -> (1, h, w, c). Here, c is the number of classification classes = 9.
    _, h, w, c = y_pred_img.shape

    # Convert predicated image (1,h,w,c) => (h,w,c)
    y_pred_img = np.reshape(y_pred_img, (h, w, c))

    img = y_pred_img
    h, w, c = img.shape

    # Generate one hot encoded array from predicted output.
    for i in range(h):
        for j in range(w):
            # Select the index of max value among the 9 classes.
            argmax_index = np.argmax(img[i, j])
            onehot_arr = np.zeros((9))
            onehot_arr[argmax_index] = 1
            img[i, j, :] = onehot_arr

    # Convert OneHotEncoded array to RGB.
    y_pred_img = onehot_to_rgb(img)

    # Get the dimensions of the original image.
    orig_img = readTiffImage4Band(fname)
    h, w, c = orig_img.shape

    # Generate image and save it.
    y_pred_img = y_pred_img[:h, :w, :]
    imx = Image.fromarray(y_pred_img)
    imx.save(
        "/home/data/IIRS/PS1-Project/src/Jun03-04/predicted/out"+str(k+1) + ".jpg")
