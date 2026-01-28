# Emotion Recognition with MobileNetV2

This project implements an emotion recognition model trained on the FER2013 dataset using a MobileNetV2-based architecture.

The model is designed to be lightweight and efficient, making it suitable for deployment on edge devices such as Raspberry Pi.

## Overview

- Emotion classification from facial images
- Transfer learning with pre-trained MobileNetV2
- Data augmentation and early stopping
- Optimized for low-resource environments

## Dataset

The model is trained on the **FER2013** dataset, which contains labeled facial expression images across multiple emotion categories.

## Model Architecture

- Base model: MobileNetV2 (ImageNet weights)
- Custom classification head:
  - Global Average Pooling
  - Dense layer with ReLU activation
  - Dropout for regularization
  - Softmax output layer

## Training Details

- Framework: TensorFlow / Keras
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam
- Data split: 80% training / 20% validation
- Early stopping based on validation loss
- Image normalization and augmentation applied

## Deployment

After training, the model was converted to a lightweight format for inference on **Raspberry Pi**, enabling real-time emotion recognition on edge devices.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- FER2013
- Raspberry Pi
