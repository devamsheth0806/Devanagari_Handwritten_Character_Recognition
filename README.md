# Devanagari Handwritten Character Recognition
## Introduction
Recognition of Devanagari characters is particularly challenging because many characters share similar shapes. Researchers have developed handcrafted features to analyze strokes for recognition, but creating a reliable feature set for Devanagari remains difficult due to the complex structure of its characters.

In this project, different machine learning algorithms were trained on the Devanagari Character dataset, and a comparative analysis was carried out among all of them. The following are the main models trained:
1. Multi-layer Perceptron (MLP): Shallow neural network
2. Naive Bayes Classifier: Probability-based machine learning classifier
3. k-Nearest Neighbors (KNN): Machine learning algorithm based on the intuitive principle that similar data points are likely to have comparable outcomes
4. Convolutional Neural Network (CNN): Training a CNN network from scratch
5. EfficientNet: A pre-trained CNN which is known for efficient computation
6. MobileNet: A pre-trained CNN specially designed for small devices
7. Deep Belief Networks (DBNs): A type of generative deep learning model composed of multiple layers of Restricted Boltzmann Machines (RBMs), stacked on top of each other

Classical machine learning models were trained on dimensionally reduced data. Principle Component Analysis (PCA) was applied to image data to find feature components with maximum variance. Components covering maximum information are considered. 

## Dataset
The Devanagari Handwritten Character Dataset (DHCD) is an extensive collection of 32x32 grayscale images that represent handwritten characters from the Devanagari script. This dataset consists of 46 distinct character classes, including 36 consonants and 10 numerals. Each class contains 2,000 samples, totaling 92,000 images. 

Dataset Link: [Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)

## Results

| **Models** |      MLP    | Naive Bayes |    KNN     |     CNN     | EfficientNet |  MobileNet  |     DBN     |
|------------|-------------|-------------|------------|-------------|--------------|-------------|-------------|
|**Validation Accuracy**| 66.60% |  54.10%  | 89.60% | 92.17% | 98.20% | 90.72% | 83.80% |

A detailed documentation can be found in this repository as [Final Report]()
