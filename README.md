#  Flower Image Classification using CNN and TensorFlow

##  Problem Statement

Develop an AI-based image classification system capable of identifying and classifying flower images into different categories using Deep Learning techniques. The project aims to implement Convolutional Neural Networks (CNNs) for image recognition and improve prediction performance using data augmentation and preprocessing techniques.

#  Project Overview

This project is a Deep Learning-based Flower Image Classification System developed using TensorFlow and Keras. The model is trained on a flower dataset containing five different flower categories:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

The system uses Convolutional Neural Networks (CNNs) to learn image patterns and classify unseen flower images with good accuracy.

The project also includes:
- Image preprocessing
- Data augmentation
- Model training and validation
- Accuracy and loss visualization
- Real-time image prediction

#  How the Project Works

1. The flower dataset is loaded using TensorFlow.
2. Images are resized and preprocessed.
3. Data augmentation techniques are applied to reduce overfitting.
4. A CNN model is created using TensorFlow/Keras.
5. The model is trained on flower images.
6. Validation accuracy and loss are calculated.
7. The trained model is saved.
8. New flower images are given as input for prediction.
9. The model predicts the flower category with confidence score.

#  Solution

The project solves the image classification problem using Convolutional Neural Networks (CNNs). CNNs automatically extract important image features such as edges, shapes, textures, and patterns from flower images.

To improve model performance and reduce overfitting, data augmentation techniques like:
- Random Flip
- Random Rotation
- Random Zoom

were implemented.

The final trained model can successfully classify unseen flower images into their correct categories.

#  Tech Stack Used

## Programming Language
- Python

## Libraries & Frameworks
- TensorFlow
- Keras
- NumPy
- Matplotlib
- pathlib

## Tools Used
- Visual Studio Code
- GitHub

#  Implementation

## Step 1 — Dataset Loading
The TensorFlow flower dataset is downloaded and loaded.

## Step 2 — Data Preprocessing
Images are resized and normalized.

## Step 3 — Data Augmentation
Random transformations are applied to improve generalization.

## Step 4 — CNN Model Creation
A Convolutional Neural Network is built using:
- Conv2D
- MaxPooling2D
- Dense Layers

## Step 5 — Model Training
The model is trained for multiple epochs using training and validation datasets.

## Step 6 — Model Evaluation
Accuracy and loss graphs are generated.

## Step 7 — Prediction System
The trained model predicts flower categories for new input images.

---

#  Project Results

- Successfully trained CNN-based flower classifier
- Achieved good validation accuracy
- Reduced overfitting using data augmentation
- Implemented real-time image prediction system


#  Conclusion

This project demonstrates the implementation of Deep Learning and Convolutional Neural Networks for image classification tasks. The model successfully classifies flower images into multiple categories using TensorFlow and Keras.

The project helped in understanding:
- CNN architecture
- Image preprocessing
- Data augmentation
- Model training and validation
- Deep Learning workflows

Future improvements can include:
- Deploying as a web application
- Increasing dataset size
- Improving accuracy with advanced CNN architectures

---

# 🔗 Project Links

## GitHub Repository
https://github.com/Pravs26/IAIP.git

## LinkedIn Link
www.linkedin.com/in/sai-pravallika-kovvuri-9b289b376

## Deployment Link
https://iaip-8gzmmiprbcphdcttyqc4cb.streamlit.app/

---

# 👨‍💻 Author

K.Sai Pravallika

---

#  Acknowledgements

- TensorFlow Official Documentation
- TensorFlow Flower Dataset
- Open Source Community
