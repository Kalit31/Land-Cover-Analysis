# Land-Cover-Analysis

This repository contains the project "Lnad Use/Land Cover change detection in cyclone affected areas using convolutional neural networks" completed at the Indian Institute of Remote Sensing, Dehradun from May 2020 to June 2020. The help in code implementation is taken from [Eye in the Sky](https://github.com/manideep2510/eye-in-the-sky).

## Project Objectives

The aim of the project is to train a convolutional neural network model for image segmentation of satellite images into the following classes: urban, rural, agriculture, deciduous forests, mangrove forests, uncultivable land, ponds/lake and canals or rivers. Upon the completion of training the model, it will then be used to detect chnages in the land cover of west bengal regions before and after the Amphan cyclone.

## Flowchart :

<p align="center">
    <img src="images/flowchart.png" height=600 />
</p>

## Testing the model's accuracy :

Here are some of the training images and the model's predicted ouptuts.

|    Satellite Images    | Predicted outputs  |     Ground Truths     |
| :--------------------: | :----------------: | :-------------------: |
| ![](images/sat-2.png)  | ![](images/2.jpg)  | ![](images/gt-2.png)  |
| ![](images/sat-14.png) | ![](images/14.jpg) | ![](images/gt-14.png) |
| ![](images/sat-22.png) | ![](images/22.jpg) | ![](images/gt-22.png) |

The dataset comprised of 18805 training images and 840 validation images, each of 128 \* 128 pixels. The model was trained for 50 epochs.
A training accuracy of about 95% was obtained and a validation accuracy of about 80%.

|            Accuracy Plot             |            Loss Plot             |
| :----------------------------------: | :------------------------------: |
| ![](images/plots/Accuracy_Plot2.png) | ![](images/plots/Loss_Plot2.png) |

Training for further 25 epochs-

|            Accuracy Plot            |            Loss Plot            |
| :---------------------------------: | :-----------------------------: |
| ![](images/plots/Accuracy_Plot.png) | ![](images/plots/Loss_Plot.png) |

## Change Detection

Some of the segmented images over a period of time.

|          April 20           |           May 10            |           May 15            |           May 30            |           June 03           |           June 09           |
| :-------------------------: | :-------------------------: | :-------------------------: | :-------------------------: | :-------------------------: | :-------------------------: |
| ![](images/apr20/out2.jpg)  | ![](images/may10/out2.jpg)  | ![](images/may15/out2.jpg)  | ![](images/may30/out2.jpg)  | ![](images/jun03/out2.jpg)  | ![](images/jun09/out2.jpg)  |
| ![](images/apr20/out16.jpg) | ![](images/may10/out16.jpg) | ![](images/may15/out16.jpg) | ![](images/may30/out16.jpg) | ![](images/jun03/out16.jpg) | ![](images/jun09/out16.jpg) |
| ![](images/apr20/out17.jpg) | ![](images/may10/out17.jpg) | ![](images/may15/out17.jpg) | ![](images/may30/out17.jpg) | ![](images/jun03/out17.jpg) | ![](images/jun09/out17.jpg) | ! |
| ![](images/apr20/out25.jpg) | ![](images/may10/out25.jpg) | ![](images/may15/out25.jpg) | ![](images/may30/out25.jpg) | ![](images/jun03/out25.jpg) | ![](images/jun09/out25.jpg) | ! |

#### For more details about the methodology and results, refer to the [project report](Report.pdf)

Here are the pre-trained weights, so that you do not have to train your model from scratch as it takes a lot many hours to process the complete dataset even on a GPU. ([download weights](https://drive.google.com/file/d/1qlv8o4Uzrd3aYhBLhbwHbHBYO1sCE62S/view?usp=sharing))
