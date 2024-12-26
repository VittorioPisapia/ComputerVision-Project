import cv2  # type: ignore
import numpy as np  # type: ignore
import os
from sklearn.model_selection import train_test_split  # type: ignore

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

sift = cv2.SIFT_create()

def extract_sift_features(images):
    descriptors_list = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors.mean(axis=0))  
        else:
            descriptors_list.append(np.zeros(128))
    return np.array(descriptors_list)

def preprocess_data_with_sift(real_dir, fake_dir, img_size=(224, 224)):

    real_images, real_labels = load_images_from_directory(real_dir, label=0, img_size=img_size)
    fake_images, fake_labels = load_images_from_directory(fake_dir, label=1, img_size=img_size)
    
    X = np.concatenate((real_images, fake_images), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)
    
    sift_features = extract_sift_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(sift_features, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
