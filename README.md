# Handwritten Digit Recognition with Keras
This repository contains code and resources for training a neural network to recognize 
handwritten digits using the MNIST dataset. The project demonstrates how to build, train, 
and evaluate a convolutional neural network (CNN) using Keras.


# Overview
Handwritten digit recognition is a classic problem in the field of machine learning and 
computer vision. The goal is to correctly identify digits (0-9) from images of handwritten numbers. 
This project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test 
images of digits written by different individuals.

# Training the Model
The mnist_train.ipynb notebook contains the code for training a neural network on the 
MNIST dataset.

# Application Notebook
The APP.ipynb notebook contains  applications of the trained model.

# Model Architecture
The model used in this project is a Convolutional Neural Network (CNN) with the following layers:

Convolutional layers with ReLU activation
MaxPooling layers for down-sampling
Flatten layer to convert 2D matrix to a 1D vector
Fully connected (Dense) layers with ReLU activation
Output layer with softmax activation for multi-class classification

# Result
The model achieves good accuracy on the MNIST test set, but sometimes it fails to recognizing hand written digits. 
Detailed training and evaluation results are provided in the mnist_train.ipynb notebook.






