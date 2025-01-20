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
### LBP
LBP is a texture descriptor that captures local features by encoding the relationship between a pixel and its neighboring pixels.
#### Advantages
- Captures fine-grained Textures: LBP could be effective in detecting irregular patterns or inconsistencies in skin texture due to its focus in local texture.
- Computational Efficiency: LBP is computationally lightweight, making it well-suited for analyzing large datasets.
- Baseline for Comparison: due to its low computational cost, it will be used as a baseline for comparison.
#### Limitations
- Struggles with high-level features, such as global structures, limiting its ability to capture complex anomalies.

