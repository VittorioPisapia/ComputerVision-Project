# Comparative Analysis of Feature Extraction Techniques for Robust Deepfake Image Detection
## Table of contents
## Introduction to the Problem
Deepfake image detection has become a critical challenge in computer vision due to the growing ease of generating highly realistic synthetic media using artificial intelligence. Detecting such manipulated content demands robust techniques capable of distinguishing between authentic and generated images. This project aims to investigate and compare various feature extraction techniques for deepfake image detection. By evaluating the performance of traditional feature extraction methods, such as Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG), alongside a neural network-based end-to-end model, the project tries to identify the strengths and weaknesses of each approach.

## Human Performance
24 participants were tested using the same dataset employed for training and evaluation. Each participant was presented with 10 images per test, and the time required to complete each test was recorded.
Out of 132 total runs, the mean accuracy was 6.59 correct answers per test, while the average time required was 49 seconds per test.
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/app1.png" alt="Example Image" style="width:660px;"/>
</p>

## Feature extractor
### LBP (Local Binary Patterns)
- **Strengths**: Efficient and great for capturing fine-grained textures, such as irregularities in skin. Used as a baseline due to its simplicity.  
- **Limitations**: Struggles with high-level patterns and global image structures.

### HOG (Histogram of Oriented Gradients)
- **Strengths**: Highlights shapes and edges, making it effective for spotting distortions in contours. Complements LBP for a richer feature set.  
- **Limitations**: Less effective in detecting subtle texture details.

### SIFT (Scale-Invariant Feature Transform)
- **Strengths**: Focuses on local keypoints, robust under scaling and rotation, ideal for anomaly detection in specific regions.  
- **Limitations**: High computational cost and large feature vectors.

### CNN (Convolutional Neural Networks)
- **Strengths**: End-to-end learning directly from raw images, extracting hierarchical features (textures, patterns, anomalies). Combines feature extraction and classification.  
- **Limitations**: Requires extensive data and computational resources. Works as a "black box," challenging to interpret.
#### Model Architecture
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/CNN_model_architecture.png" alt="Example Image" style="width:660px;"/>
</p>

## Classification
Once the features have been acquired, classification will be performed by SVM model trained for each specific feature extractor (or combination of feature extractor).
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/classification.png" alt="Example Image" style="width:660px;"/>
</p>

## Dataset
The dataset for this project includes both real and fake images:
- Real images: Samples were sourced from Flickr-Faces-HQ (FFHQ), provided by Nvidia.
- Fake images: Faces were sampled by the “1 Million Face Faces” dataset, created with StyleGan and provided by Bojan Tunguz.

## Evaluation Metrics 
- **Accuracy**: Evaluated using the F1-score to measure the balance between precision and recall in the classification.
- **Computational Efficiency**: Feature extraction times and training times will be compared for each method to assess computational performance.
- **Robustness to Adversarial Attacks**: Test on adversarial examples generated with Fast Gradient Method (FGM) and Carlini and Wagner L2 Method (CL2).




