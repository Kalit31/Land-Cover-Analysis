# Sentinel-2 MSI: MultiSpectral Instrument, Level-1C Segmentation

## Testing the model's accuracy :

Here are some of the training images and the model's predicted ouptuts.

|    Satellite Images    | Predicted outputs  |     Ground Truths     |
| :--------------------: | :----------------: | :-------------------: |
| ![](images/sat-2.png)  | ![](images/2.jpg)  | ![](images/gt-2.png)  |
| ![](images/sat-14.png) | ![](images/14.jpg) | ![](images/gt-14.png) |
| ![](images/sat-22.png) | ![](images/22.jpg) | ![](images/gt-22.png) |

## Plots

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

Here are the pre-trained weights, ([download weights](https://drive.google.com/file/d/1qlv8o4Uzrd3aYhBLhbwHbHBYO1sCE62S/view?usp=sharing))
