import numpy as np # type: ignore
from skimage.feature import hog # type: ignore
from sklearn.svm import SVC # type: ignore
from art.attacks.evasion import CarliniL2Method, FastGradientMethod # type: ignore
from art.estimators.classification import SklearnClassifier # type: ignore
import pickle
from data_preprocessing import preprocess_data_with_hog
from numpy.linalg import norm # type: ignore

# Carica il modello salvato
print("Loading the model...")
with open('D:\CV\Models_and_scalers\hog_svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('D:\CV\Models_and_scalers\scaler_HOG.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Definisci il classificatore ART
classifier = SklearnClassifier(model=svm)

# Preprocessa i dati per ottenere il test set
_, X_test, _, y_test = preprocess_data_with_hog('D:/CV/dataset/0', 'D:/CV/dataset/1')

# Prendi un campione di feature HOG
hog_feature = X_test[0]

# Assicurati che HOG feature abbia la giusta forma (bidimensionale)
hog_feature = hog_feature.reshape(1, -1)  # Deve avere forma (1, n_features)

# Scala la feature originale prima di generare l'attacco
hog_feature_scaled = scaler.transform(hog_feature)

# Definisci l'attacco adversariale
attack = FastGradientMethod(estimator=classifier, eps=0.03)

# Genera feature perturbate
print("Generating adversarial features...")
adversarial_features_scaled = attack.generate(x=hog_feature_scaled)

# Predizioni sul modello originale e con feature avversarie
original_prediction = svm.predict(hog_feature_scaled)
adversarial_prediction = svm.predict(adversarial_features_scaled)

# Probabilità delle predizioni
original_proba = svm.predict_proba(hog_feature_scaled)
adversarial_proba = svm.predict_proba(adversarial_features_scaled)

# Misura della differenza tra le feature
feature_difference = norm(hog_feature_scaled - adversarial_features_scaled)
original_norm = norm(hog_feature_scaled)
adversarial_norm = norm(adversarial_features_scaled)

# Stampa dei risultati
print(f"Original Prediction: {original_prediction}, Probability: {original_proba}")
print(f"Adversarial Prediction: {adversarial_prediction}, Probability: {adversarial_proba}")
print(f"Feature Difference (L2 norm): {feature_difference}")
print(f"original_norm (L2 norm): {original_norm}")
print(f"adversarial_norm (L2 norm): {adversarial_norm}")

