from osgeo import gdal, osr
import numpy as np
#from image_processing_functions import *
from unet_model import UNet_1Band
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Remove the comment if using pretrained weights
# Set the path appropriately
#weights_file = "/home/data/IIRS/December-Testing/TrainedModels/trained_model_sar-1.h5"


###
trainx_cropped = np.load(
    '/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/NumpyArrays/trainX-sar.npy')
print("TrainX Images: "+str(trainx_cropped.shape))
###
trainy_hot = np.load(
    '/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/NumpyArrays/trainYHot-sar.npy')
print("TrainY Images: "+str(trainy_hot.shape))
###
validationx_cropped = np.load(
    '/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/NumpyArrays/trainXVal-sar.npy')
print("Validation X Images: "+str(validationx_cropped.shape))
###
validationy_hot = np.load(
    '/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/NumpyArrays/trainYHotVal-sar.npy')
print("Validation Y Images: "+str(validationy_hot.shape))
###

print('--------------Loaded Data----------------')

# Import Unet model for SAR segmentation
model = UNet_1Band()

# Load trained weights
# Remove the comment if further training required
# model.load_weights(weights_file)

# Train the model
history = model.fit(trainx_cropped, trainy_hot, validation_data=(
    validationx_cropped, validationy_hot), epochs=30, batch_size=16, verbose=1)

# Save the model
model.save("/home/data/IIRS/December-Testing/Sentinel-1-SAR:1-band/TrainedModels/trained_model_sar-v2.h5")

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
plt.savefig('Accuracy_Plot-SAR.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('Loss_Plot-SAR.png')
