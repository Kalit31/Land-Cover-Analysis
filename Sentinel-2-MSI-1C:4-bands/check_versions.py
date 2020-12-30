from tensorflow.python.client import device_lib
import re
from osgeo import gdal, osr
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import glob
import cv2
import os
import math
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

print("Tensorflow :", tf.__version__)
print("Numpy: ", np.__version__)

print(device_lib.list_local_devices())
