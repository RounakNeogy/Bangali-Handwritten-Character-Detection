# Bengali Handwritten Character Recognition

## Introduction
This project focuses on developing a Convolutional Neural Network (CNN) model for the recognition of handwritten Bengali characters. The ability to accurately identify handwritten characters is crucial for various applications, including optical character recognition (OCR), language processing, and document digitization. With the rise of digital platforms and the increasing need for automated systems, efficient character recognition algorithms play a significant role in bridging the gap between handwritten content and digital processing.

## Description
The Bengali Handwritten Character Recognition project employs Python programming language along with deep learning libraries such as Keras and TensorFlow to build and train a CNN model. The model is trained on the CMATERdb dataset and BanglaLekha-Isolated dataset, widely used benchmark datasets for Bengali handwritten character recognition tasks.

## Key Features
- Utilizes Convolutional Neural Network architecture for feature extraction and classification.
- Implemented in Python for flexibility, ease of use, and extensive libraries support.
- Leveraged Keras, a high-level neural networks API, for building and training the model.
- Employs TensorFlow as the backend for efficient computation and optimization.
- Achieves an impressive accuracy of 96.37% on the CMATERdb Dataset and 93.17% on the BanglaLekha-Isolated dataset demonstrating the robustness and effectiveness of the developed model.

## Image Processing
Function named ```process_images()``` that processes images in the specified folder, iterates through each sub-folder, loads images, resizes them, converts to grayscale, and applies Otsu's thresholding for binary image conversion. The function distinguishes between training and test folders, adjusting the processing based on folder type. It prints the progress of image processing and the total number of images processed. This function is called twice, once for the training folder and once for the test folder, to prepare images for further processing in a Bengali handwritten character recognition system.

## Model
The model is a Convolutional Neural Network (CNN) architecture augmented with a Long Short-Term Memory (LSTM) layer for Bengali handwritten character recognition. Initially, the CNN component consists of three convolutional layers, each followed by Leaky ReLU activation functions for introducing non-linearity, BatchNormalization for stabilizing training, and Dropout layers to prevent overfitting. MaxPooling layers reduce spatial dimensions, aiding in feature extraction. Subsequently, the Reshape layer prepares the data for input into the LSTM, which specializes in processing sequential data. The LSTM layer with 128 units retains memory of previous inputs, enhancing the model's ability to recognize sequential patterns in handwritten characters. Following the LSTM layer, the data is flattened and passed through fully connected layers with Leaky ReLU activations and Dropout regularization. Finally, the model is compiled using the Adadelta optimizer with a categorical cross-entropy loss function to classify Bengali characters among 50 classes with high accuracy.

## Dataset
### CMATERdb 
CMATERdb is a benchmark dataset for Bengali handwritten character recognition. It contains diverse handwritten characters for training and evaluation. The dataset is split into:
- Training Set (10800 images): Used to train the model.
- Validation Set (1200 images): Used for hyperparameter tuning and performance monitoring.
- Test Set (3000 images): Used to evaluate the final model performance.

### BanglaLekha-Isolated
The BanglaLekha-Isolated dataset comprises 98,950 images categorized into 50 classes, utilized for Bengali character recognition tasks. The dataset is split into three subsets:
- Training Set (69,244 images): Used to train the recognition model, providing diverse examples for learning character patterns.
- Validation Set (9,877 images): Reserved for fine-tuning model hyperparameters and monitoring training performance.
- Test Set (19,829 images): Independently evaluated to assess the model's accuracy and generalization on unseen data.

