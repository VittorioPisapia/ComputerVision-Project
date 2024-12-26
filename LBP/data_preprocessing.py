import cv2  # type: ignore
import numpy as np  # type: ignore
from skimage.feature import local_binary_pattern  # type: ignore
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

def extract_lbp_features(images, radius=3, n_points=None):
    if n_points is None:
        n_points = 8 * radius 

    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= hist.sum()  
        features.append(hist)
    
    return np.array(features)

def preprocess_data_with_lbp(real_dir, fake_dir, img_size=(224, 224)):
    real_images, real_labels = load_images_from_directory(real_dir, label=0, img_size=img_size)
    fake_images, fake_labels = load_images_from_directory(fake_dir, label=1, img_size=img_size)
    
    X = np.concatenate((real_images, fake_images), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)
    
    lbp_features = extract_lbp_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(lbp_features, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
