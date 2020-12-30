from osgeo import gdal, osr
import numpy as np
from RGBOneHotConv import *


# Convert a 1band-TIFF image to numpy array using gdal
def readTiffImage1Band(fname):
    tif = gdal.Open(fname)
    img = tif.ReadAsArray()
    return img


# Convert a 4band-TIFF image to numpy array using gdal
def readTiffImage4Band(fname):
    tif = gdal.Open(fname)
    img = tif.ReadAsArray()
    img = np.stack((img[0], img[1], img[2], img[3]), axis=-1)
    return img


# Padding at the bottom and at the left of images to be able to crop them into 128*128 images for training
def padding(img, w, h, crop_size, stride, n_h, n_w, c=None):

    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    if(c == None):
        img_pad = np.pad(
            img, [(0, h_toadd), (0, w_toadd)], mode='constant')
    else:
        img_pad = np.pad(
            img, [(0, h_toadd), (0, w_toadd), (0, 0)], mode='constant')

    return img_pad


# Adding pixels to make the image with shape in multiples of stride(32)
# The pixels to add in the height come from the initial pixels
# Similarly, for the width
def add_pixels(img, h, w, n_h, n_w, crop_size, stride, c=None):

    # make width multiple of 32
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    # make height multiple of 32
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    if(c == None):
        # initialize a numpy array with new width and height
        img_add = np.zeros(((h+h_toadd), (w+w_toadd)))
        # copy the original image
        img_add[:h, :w] = img
        # copy the required top rows to the bottom rows in new array
        img_add[h:, :w] = img[:h_toadd, :]
        # copy the required leftmost columns to the rightmost in new array
        img_add[:h, w:] = img[:, :w_toadd]
        # copy the remaining square from the orig img
        img_add[h:, w:] = img[h-h_toadd:h, w-w_toadd:w]
    else:
        # initialize a numpy array with new width and height
        img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
        # copy the original image
        img_add[:h, :w, :] = img
        # copy the required top rows to the bottom rows in new array
        img_add[h:, :w, :] = img[:h_toadd, :, :]
        # copy the required leftmost columns to the rightmost in new array
        img_add[:h, w:, :] = img[:, :w_toadd, :]
        # copy the remaining square from the orig img
        img_add[h:, w:, :] = img[h-h_toadd:h, w-w_toadd:w, :]

    return img_add


# Slicing the image into crop_size*crop_size crops
def crops(a, crop_size=128):

    stride = 32

    croped_images = []

    # 1 Band
    if(len(a.shape) == 2):
        h, w = a.shape

        n_h = int(int(h/stride))
        n_w = int(int(w/stride))

        # Adding pixals as required
        a = add_pixels(a, h, w, n_h, n_w, crop_size, stride)

        # Slicing the image into 128*128 crops with a stride of 32
        for i in range(n_h-1):
            for j in range(n_w-1):
                crop_x = a[(i*stride):((i*stride)+crop_size),
                           (j*stride):((j*stride)+crop_size)]
                crop_x = crop_x[..., np.newaxis]
                croped_images.append(crop_x)
    else:
        # 4 Bands

        h, w, c = a.shape
        n_h = int(int(h/stride))
        n_w = int(int(w/stride))

        # Adding pixals as required
        a = add_pixels(a, h, w, c, n_h, n_w, crop_size, stride)

        # Slicing the image into 128*128 crops with a stride of 32
        for i in range(n_h-1):
            for j in range(n_w-1):
                crop_x = a[(i*stride):((i*stride)+crop_size),
                           (j*stride):((j*stride)+crop_size), :]
                croped_images.append(crop_x)

    return croped_images


def getTrainingCroppedImages1Band(filelist_train):
    train_list_cropped = []

    for fname in filelist_train:
        # Reading the image
        image = readTiffImage1Band(fname)

        # Padding as required and cropping
        crops_list = crops(image)

        train_list_cropped = train_list_cropped + crops_list

    # Array of all the cropped Training sat Images
    train_cropped = np.asarray(train_list_cropped)
    del train_list_cropped
    return train_cropped


def getTrainingCroppedImagesMorethan1Band(filelist_train):
    train_list_cropped = []

    for fname in filelist_train:
        # Reading the image
        image = readTiffImage1Band(fname)

        # Padding as required and cropping
        crops_list = crops(image)

        train_list_cropped = train_list_cropped + crops_list

    # Array of all the cropped Training sat Images
    train_cropped = np.asarray(train_list_cropped)
    del train_list_cropped
    return train_cropped


def convertRGBtoOneHot(trainy_cropped):
    trainy_hot = []
    for i in range(trainy_cropped.shape[0]):
        hot_img = rgb_to_onehot(trainy_cropped[i, :, :, :3])
        trainy_hot.append(hot_img)

    trainy_hot = np.asarray(trainy_hot)
    return trainy_hot
