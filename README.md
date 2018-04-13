# CNN_IMU
Implementation code and parameters of the CNN-IMU presented in the Journal "Convolutional Neural Networks for Human Activity Recognition using Body-Worn Sensors"

Updating in progress.

## Abstract
Abstract: Human activity recognition (HAR) is a classification task for recognizing human movements. Methods of HAR are of great interest as they became a tool for measuring occurrences and durations of human actions, which are the basis of smart assistive technologies and manual processes analysis. Recently, deep neural networks have been deployed for HAR in the context
of activities of daily living using multichannel time-series. These time-series are acquired from body-worn devices, which are composed of different types of sensors. The deep architectures process these measurements for finding basic and complex features in human corporal movements, and for classifying them into a set of human actions. As the devices are worn at different parts of the human body, we propose a novel deep neural network for HAR. This network handles sequence measurements from different body-worn devices separately. An evaluation of the architecture is performed on three datasets, the Oportunity, Pamap2, and an industrial dataset, outperforming the state-of-the-art. In addition, different network configurations will also be evaluated. We find that applying convolutions per sensor channe

## Prerequisites
The implementation of the CNN-IMU has the following dependencies:
- Caffe
- numpy
- OpenCV/Python


## For testing
Protoxtx files and caffemodels (weights for caffe) are found in CNN_IMU_rep

run main.py selecting the dataset and the network type.


## Networks 
Networks, their configurations per dataset and the solver could be created using CNN_IMU_rep/src/CNN.py
- For testing, each dataset contains the prototxt file of the architectures and some (due to storage limits) caffemodels.

## Dataset
Pamap2 preprocessing found in CNN_IMU_rep/src/preprocessing_pamap2.py
