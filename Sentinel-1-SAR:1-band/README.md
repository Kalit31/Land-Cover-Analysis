# Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected Segmentation

## Testing the model's accuracy :

Here are some of the training images and the model's predicted ouptuts.

|    Satellite Images    | Predicted outputs  |     Ground Truths     |
| :--------------------: | :----------------: | :-------------------: |
| <img src="images/2.png" alt="2.png" width="200"/>   | <img src="images/out2.jpg" alt="out2.jpg" width="200"/>  | <img src="images/gt-2.png" alt="gt-2.png" width="200"/>  |
| <img src="images/18.png" alt="18.png" width="200"/>| <img src="images/out18.jpg" alt="out18.jpg" width="200"/> | <img src="images/gt-18.png" alt="gt-18.png" width="200"/>  |
| <img src="images/21.png" alt="21.png" width="200"/> | <img src="images/out21.jpg" alt="out21.jpg" width="200"/> | <img src="images/gt-21.png" alt="gt-21.png" width="200"/>  |

## Plots

The dataset comprised of 17844 training images and 1801 validation images, each of 128 \* 128 pixels. The model was trained for 30 epochs.
A training accuracy of about 93% was obtained and a validation accuracy of about 65%.

|            Accuracy Plot             |            Loss Plot             |
| :----------------------------------: | :------------------------------: |
| ![](images/plots/Accuracy_Plot-SAR-2.png) | ![](images/plots/Loss_Plot-SAR-2.png) |

## Change Detection

Some of the segmented images over a period of time.

|          Jan-March          |           April-June           |           July-Sept            |           Oct-Dec            |
| :-------------------------: | :-------------------------: | :-------------------------: | :-------------------------: |
| ![](images/Jan-March/out1.jpg)  | ![](images/Apr-June/out1.jpg)  | ![](images/July-Sept/out1.jpg)  | ![](images/Oct-Dec/out1.jpg) | 
| ![](images/Jan-March/out3.jpg) | ![](images/Apr-June/out3.jpg) | ![](images/July-Sept/out3.jpg) | ![](images/Oct-Dec/out3.jpg)|
| ![](images/Jan-March/out11.jpg) |  ![](images/Apr-June/out11.jpg) | ![](images/July-Sept/out11.jpg)| ![](images/Oct-Dec/out11.jpg)|
| ![](images/Jan-March/out14.jpg) |  ![](images/Apr-June/out14.jpg)| ![](images/July-Sept/out14.jpg) | ![](images/Oct-Dec/out14.jpg)|
| ![](images/Jan-March/out22.jpg) |  ![](images/Apr-June/out22.jpg)| ![](images/July-Sept/out22.jpg) | ![](images/Oct-Dec/out22.jpg)|

Here are the pre-trained weights, ([download weights](https://drive.google.com/file/d/13dyzByX94_a8iqIy7UBuSI3eIAMuxPqp/view?usp=sharing))
