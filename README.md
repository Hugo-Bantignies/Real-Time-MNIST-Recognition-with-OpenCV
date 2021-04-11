# Real-Time-MNIST-Recognition-with-OpenCV

## Description

This project is an initiation to Deep Learning and Neural Networks using Keras. With a Convolutional Neural Network (CNN) following the LeNet5 architecture, this application recognize handwritten digits in real time.

### Convolutional Neural Network

The CNN was trained with ***the MNIST Dataset from Yann Lecun*** (http://yann.lecun.com/exdb/mnist/).
It follows the LeNet5 which is one very famous architecture of Convolutional Neural Network. 

### OpenCV application for realtime recognition

We will load the trained model and using OpenCV, we can recognize digits by prediction from the model. To detect digits, there is a rectangle applied on the webcam. We binarize with a threshold the region within the rectangle to get the digit. 

