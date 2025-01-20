# Comparative Analysis of Feature Extraction Techniques for Robust Deepfake Image Detection
## Table of Contents
- [Introduction to the Problem](#introduction-to-the-problem)
- [Human Performance](#human-performance)
- [Feature Extractor](#feature-extractor)
- [Classification](#classification)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Feature Extractor Performance](#feature-extractor-performance)
- [Adversarial Robustness](#adversarial-robustness)
- [Conclusion](#conclusion)
## Introduction to the Problem
Detecting deepfake images is a pressing challenge in computer vision, given the ease of generating hyper-realistic synthetic media with AI. This project explores and compares different feature extraction techniques to tackle this issue. By analyzing methods like Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), and an end-to-end neural network, the goal is to identify their strengths and weaknesses in distinguishing authentic from fake images.
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/gif_fake_faces.gif" alt="Example Image" style="width:500px;"/>
</p>

## Human Performance
To assess human ability in detecting deepfakes, 24 participants were tested with the same dataset used for training and evaluation. Each participant analyzed 10 images per test, with results showing:
- **Mean accuracy**: 6.59 correct answers per test.
- **Average time**: 49 seconds per test.

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
This project explores two SIFT implementations:
1. **Dense SIFT**: Focuses on predetermined points arranged on a grid, providing a uniform analysis of the image.
2. **Mean SIFT Descriptors**: Aggregates the mean of all SIFT descriptors found in a given image, offering a compact representation.

- **Strengths**: Robust to scaling and rotation, excels at detecting local anomalies, 128 component vector (Mean SIFT Descriptors).  
- **Limitations**: Computationally expensive, with high-dimensional feature vectors (Dense SIFT).

### CNN (Convolutional Neural Networks)
- **Strengths**: End-to-end learning directly from raw images, extracting hierarchical features (textures, patterns, anomalies). Combines feature extraction and classification.  
- **Limitations**: Requires extensive data and computational resources. Works as a "black box," challenging to interpret.
#### Model Architecture
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/CNN_model_architecture.png" alt="Example Image" style="width:500px;"/>
</p>

## Classification
Once the features have been acquired, classification will be performed by SVM model trained for each specific feature extractor (or combination of feature extractor).
<p align="center">
    <img src="https://github.com/VittorioPisapia/ComputerVision-Project/blob/main/images/classification.png" alt="Example Image" style="width:500px;"/>
</p>

## Dataset
The dataset for this project includes both real and fake images:
- Real images: Samples were sourced from Flickr-Faces-HQ (FFHQ), provided by Nvidia.
- Fake images: Faces were sampled by the “1 Million Face Faces” dataset, created with StyleGan and provided by Bojan Tunguz.

## Evaluation Metrics 
- **Accuracy**: Evaluated using the F1-score to measure the balance between precision and recall in the classification.
- **Computational Efficiency**: Feature extraction times and training times will be compared for each method to assess computational performance.
- **Robustness to Adversarial Attacks**: Test on adversarial examples generated with Fast Gradient Method (FGM) and Carlini and Wagner L2 Method (CL2).

## Feature Extractor Performance

| Feature Extractor | Training Time (seconds) | Accuracy | F1-Score |
|--------------------|--------------------------|----------|----------|
| LBP               | 86.17                   | 0.64     | 0.64     |
| HOG               | 67.97                   | 0.74     | 0.74     |
| LBP + HOG         | 1908.29                 | 0.82     | 0.82     |
| SIFT              | 47.84                   | 0.62     | 0.62     |
| SIFT + HOG        | 2125.73                 | 0.81     | 0.81     |
| CNN (20 epochs)   | 919.38                  | 0.69     | 0.69     |
| CNN (10 epochs)   | 456.57                  | 0.70     | 0.70     |
| CNN (5 epochs)    | 232.03                  | 0.67     | 0.66     |

## Adversarial Robustness
Adversarial robustness was evaluated using two attacks from the Adversarial Robustness Toolbox (ART):

### Attack Methods
- **Fast Gradient Method (FGM)**: An extension of the Fast Gradient Sign Method. FGM computes the gradient of the loss function with respect to the input features and generates adversarial examples by perturbing the input in the direction that maximizes the loss. It is computationally efficient and straightforward.
- **Carlini and Wagner L2 Attack (CL2)**: A state-of-the-art iterative method that formulates adversarial example generation as an optimization problem. It seeks the minimal perturbation required to misclassify the model, making it more precise and effective compared to FGM.

---

#### LBP
| Attack           | Mean of Original L2 Norms | Mean of L2 Norm Differences | % w.r.t. Original Norms | Successful Attacks (%) |
|-------------------|---------------------------|-----------------------------|--------------------------|-------------------------|
| FGM (ε = 0.01)   | 4.7768                    | 0.051                       | 1.067                   | 5                       |
| FGM (ε = 0.03)   | 4.7768                    | 0.153                       | 3.2                     | 15                      |
| FGM (ε = 0.08)   | 4.7768                    | 0.4079                      | 8.53                    | 49                      |
| CL2 (c = 0.05)   | 4.7768                    | 0.5024                      | 10.51                   | 95                      |
| CL2 (c = 0.08)   | 4.7768                    | 0.5491                      | 11.49                   | 95                      |
| CL2 (c = 0.1)    | 4.7768                    | 0.5695                      | 11.92                   | 95                      |


---

#### HOG
| Attack           | Mean of Original L2 Norms | Mean of L2 Norm Differences | % w.r.t. Original Norms | Successful Attacks (%) |
|-------------------|---------------------------|-----------------------------|--------------------------|-------------------------|
| FGM (ε = 0.01)   | 89.7785                   | 0.9                         | 1.002                   | 22                      |
| FGM (ε = 0.03)   | 89.7785                   | 2.7                         | 3.007                   | 73                      |
| FGM (ε = 0.08)   | 89.7785                   | 7.2                         | 8.019                   | 100                     |
| CL2 (c = 0.05)   | 89.7785                   | 0.3745                      | 0.4171                  | 41                      |
| CL2 (c = 0.08)   | 89.7785                   | 0.3963                      | 0.4414                  | 40                      |
| CL2 (c = 0.1)    | 89.7785                   | 0.3464                      | 0.3858                  | 36                      |


---

#### LBP + HOG
| Attack           | Mean of Original L2 Norms | Mean of L2 Norm Differences | % w.r.t. Original Norms | Successful Attacks (%) |
|-------------------|---------------------------|-----------------------------|--------------------------|-------------------------|
| FGM (ε = 0.01)   | 162.3967                  | 1.6208                      | 0.998                   | 24                      |
| FGM (ε = 0.03)   | 162.3967                  | 4.8624                      | 2.9941                  | 77                      |
| FGM (ε = 0.08)   | 162.3967                  | 12.9664                     | 7.9843                  | 99                      |

---

#### SIFT
| Attack           | Mean of Original L2 Norms | Mean of L2 Norm Differences | % w.r.t. Original Norms | Successful Attacks (%) |
|-------------------|---------------------------|-----------------------------|--------------------------|-------------------------|
| FGM (ε = 0.01)   | 11.0991                   | 0.1131                      | 1.019                   | 16                      |
| FGM (ε = 0.03)   | 11.0991                   | 0.3394                      | 3.057                   | 45                      |
| FGM (ε = 0.08)   | 11.0991                   | 0.9051                      | 8.1547                  | 89                      |
| CL2 (c = 0.05)   | 11.0991                   | 0.5006                      | 4.5102                  | 95                      |
| CL2 (c = 0.08)   | 11.0991                   | 0.5656                      | 5.0959                  | 95                      |
| CL2 (c = 0.1)    | 11.0991                   | 0.6084                      | 5.4815                  | 95                      |


---

#### SIFT + HOG
| Attack           | Mean of Original L2 Norms | Mean of L2 Norm Differences | % w.r.t. Original Norms | Successful Attacks (%) |
|-------------------|---------------------------|-----------------------------|--------------------------|-------------------------|
| FGM (ε = 0.01)   | 162.7266                  | 1.6239                      | 0.9979                  | 20                      |
| FGM (ε = 0.03)   | 162.7266                  | 4.8718                      | 2.9938                  | 76                      |
| FGM (ε = 0.08)   | 162.7266                  | 12.9916                     | 7.9836                  | 98                      |


---

#### CNN (10 Epochs)
| Attack           | Accuracy on Benign Samples | Accuracy on Adversarial Samples | Success Rate (%) |
|-------------------|----------------------------|----------------------------------|------------------|
| FGM (ε = 0.01)   | 0.7                        | 0.59                            | 41               |
| FGM (ε = 0.03)   | 0.74                       | 0.54                            | 58               |
| FGM (ε = 0.08)   | 0.71                       | 0.54                            | 58               |

--- 

## Conclusion

### Key Takeaways
- **Top Performance**: The best accuracy and F1-score were achieved by combining traditional features (LBP + HOG). However, this came at a cost:
  - Training time increased dramatically (~2000% of LBP’s training time).
  - The model was highly vulnerable to adversarial attacks, failing 99% of the time under an FGM attack (ϵ = 0.08), despite L2 norm differences staying below 10%.

- **Balanced Performer**: Among traditional methods, HOG stood out for its balance between performance and efficiency:
  - **Training Time**: Required only ~68 seconds, 78.8% of LBP’s training time.
  - **Accuracy**: Outperformed human participants, higher than LBP, CNN and SIFT.
  - **Adversarial Robustness**: Comparable to the LBP + HOG model against simpler attacks, with strong resilience against more advanced attacks (success rates below 45% for tested confidence levels).

- **Potential of CNNs**: While CNN models are known to achieve superior accuracy with sufficient resources and larger datasets, their performance in this study was limited by computational constraints:
  - **Robustness**: CNNs demonstrated better resistance to FGM attacks, maintaining strong performance even at higher ϵ values.
  - **Trade-off**: Despite their robustness, CNNs underperformed in accuracy compared to concatenated traditional features.

