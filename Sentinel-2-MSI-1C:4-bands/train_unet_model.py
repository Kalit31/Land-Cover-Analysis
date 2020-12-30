from osgeo import gdal, osr
import numpy as np
#from image_processing_functions import *
from unet_model import UNet
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#weights_file = "/home/data/IIRS/December-Testing/TrainedModels/trained_model_sar-1.h5"


###
trainx_cropped = np.load(
    '/home/data/IIRS/December-Testing/NumpyArrays-1/trainX-sar.npy')
print("TrainX Images: "+str(trainx_cropped.shape))
###
trainy_hot = np.load(
    '/home/data/IIRS/December-Testing/NumpyArrays-1/trainYHot-sar.npy')
print("TrainY Images: "+str(trainy_hot.shape))
###
validationx_cropped = np.load(
    '/home/data/IIRS/December-Testing/NumpyArrays-1/trainXVal-sar.npy')
print("Validation X Images: "+str(validationx_cropped.shape))
###
validationy_hot = np.load(
    '/home/data/IIRS/December-Testing/NumpyArrays-1/trainYHotVal-sar.npy')
print("Validation Y Images: "+str(validationy_hot.shape))
###

print('--------------Loaded Data----------------')

# Import Unet model
model = UNet()

# Load trained weights
# model.load_weights(weights_file)

# Train the model
history = model.fit(trainx_cropped, trainy_hot, validation_data=(
    validationx_cropped, validationy_hot), epochs=30, batch_size=16, verbose=1)

# Save the model
model.save("/home/data/IIRS/December-Testing/TrainedModels/trained_model_sar-v2.h5")

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
plt.savefig('Accuracy_Plot-SAR-2.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('Loss_Plot-SAR-2.png')
