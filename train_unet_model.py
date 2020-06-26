from osgeo import gdal, osr
import numpy as np
from image_processing_functions import *
from unet_model import UNet
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####
trainx_cropped = np.load('/home/data/IIRS/PS1-Project/src/trainX.npy')
####
trainy_hot = np.load('/home/data/IIRS/PS1-Project/src/trainYHot.npy')
####
validationx_cropped = np.load('/home/data/IIRS/PS1-Project/src/trainXVal.npy')
####
validationy_hot = np.load(
    '/home/data/IIRS/PS1-Project/src/trainYValHot.npy')
####

print('--------------Loaded Data----------------')

# Import Unet model
model = UNet()


# Train the model
history = model.fit(trainx_cropped, trainy_hot, validation_data=(
    validationx_cropped, validationy_hot), epochs=50, batch_size=16, verbose=1)

# Save the model
model.save("/home/data/IIRS/PS1-Project/src/trained_model.h5")

print('--------------Training Completed----------------')

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('Accuracy_Plot.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('Loss_Plot.png')
