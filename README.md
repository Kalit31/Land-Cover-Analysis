# Land-Cover-Analysis

This repository contains the project "Land Use/Land Cover change detection in cyclone affected areas using convolutional neural networks" completed at the Indian Institute of Remote Sensing, Dehradun from May 2020 to June 2020. Help in code implementation is taken from [Eye in the Sky](https://github.com/manideep2510/eye-in-the-sky).

## Project Objectives

The aim of the project is to train a convolutional neural network model for image segmentation of satellite images into the following classes: urban, rural, agriculture, deciduous forests, mangrove forests, uncultivable land, ponds/lake and canals or rivers. Upon the completion of training the model, it will then be used to detect changes in the land cover of west bengal regions before and after the Amphan cyclone.

## Flowchart :

<p align="center">
    <img src="images/flowchart.png" height=600 />
</p>


### The UNET model was trained upon two types of satellite images:

* [Sentinel-2-MSI-1C](Sentinel-2-MSI-1C:4-bands)  
* [Sentinel-1-SAR](Sentinel-1-SAR:1-band)  

#### For more details about the methodology and results, refer to the [project report](Report.pdf)

### Regarding datasets:
* Satellite images were extracted from Google Earth Engine for both the Sentinel-1 and Sentinel-2 satellites.
Here is the link for it: https://code.earthengine.google.com/. The script for this is added to the repository.
* For the ground truths, the Bhuvan platform was used. Here is the link: https://bhuvan-app1.nrsc.gov.in/thematic/thematic/index.php
* My dataset link: https://drive.google.com/drive/u/1/folders/1ZMlwmboKkTi8bayKyeEBa1XTiHCbLtBa (This is my university's google drive which may get removed soon). Though, the data here is not kept in a structured way.
