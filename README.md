# Land-Cover-Analysis

This repository contains the project "Land Use/Land Cover change detection in cyclone affected areas using convolutional neural networks" completed at the Indian Institute of Remote Sensing, Dehradun from May 2020 to June 2020. The help in code implementation is taken from [Eye in the Sky](https://github.com/manideep2510/eye-in-the-sky).

## Project Objectives

The aim of the project is to train a convolutional neural network model for image segmentation of satellite images into the following classes: urban, rural, agriculture, deciduous forests, mangrove forests, uncultivable land, ponds/lake and canals or rivers. Upon the completion of training the model, it will then be used to detect chnages in the land cover of west bengal regions before and after the Amphan cyclone.

## Flowchart :

<p align="center">
    <img src="images/flowchart.png" height=600 />
</p>


### The UNET model was trained upon two types of satellite images:

* [Sentinel-2-MSI-1C](Sentinel-2-MSI-1C:4-bands)  
* [Sentinel-1-SAR](Sentinel-1-SAR:1-band)  

#### For more details about the methodology and results, refer to the [project report](Report.pdf)
