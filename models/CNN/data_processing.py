import cv2 # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import os

def load_images_from_directory(directory, label, img_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data_cnn(real_dir, fake_dir):
    real_images, real_labels = load_images_from_directory(real_dir, label=0)
    fake_images, fake_labels = load_images_from_directory(fake_dir, label=1)
    print('Images correctly loaded')
    
    X = np.concatenate((real_images, fake_images), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)
    
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set and Testing set generated')
    
    return X_train, X_test, y_train, y_test