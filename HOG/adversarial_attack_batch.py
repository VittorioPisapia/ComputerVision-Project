import numpy as np # type: ignore
from sklearn.svm import SVC # type: ignore
from art.attacks.evasion import CarliniL2Method, FastGradientMethod,ProjectedGradientDescent # type: ignore
from art.estimators.classification import SklearnClassifier # type: ignore
import pickle
from data_preprocessing import preprocess_data_with_hog
from numpy.linalg import norm # type: ignore

# Loading Model and Scaler
print("Loading the model...")
with open('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\hog_svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\scaler_HOG.pkl', 'rb') as f:
    scaler = pickle.load(f)

classifier = SklearnClassifier(model=svm)
_, X_test, _, y_test = preprocess_data_with_hog('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\dataset\\0', 'C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\dataset\\1')

batch_size = 100
X_batch = X_test[:batch_size]
y_batch = y_test[:batch_size]

X_batch_scaled = scaler.transform(X_batch)

# Choose attack Method 
#attack = FastGradientMethod(estimator=classifier, eps=0.01)
#attack = ProjectedGradientDescent(estimator=classifier, eps=0.03, max_iter=20)
attack = CarliniL2Method(classifier=classifier, confidence=0.1, max_iter=20)

print(f"Generating adversarial features for a batch of {batch_size} samples...")
adversarial_features_scaled = attack.generate(x=X_batch_scaled)

original_predictions = svm.predict(X_batch_scaled)
adversarial_predictions = svm.predict(adversarial_features_scaled)

successful_attacks = np.sum(original_predictions != adversarial_predictions)

l2_differences = [norm(original - adversarial) 
                  for original, adversarial in zip(X_batch_scaled, adversarial_features_scaled)]

original_norms = [norm(original) for original in X_batch_scaled]
mean_original_norm = np.mean(original_norms)

mean_l2_difference = np.mean(l2_differences)

print(f"Original Predictions: {original_predictions}")
print(f"Adversarial Predictions: {adversarial_predictions}")
print(f"Number of successful attacks: {successful_attacks} out of {batch_size}")
print(f"Mean L2 norm of differences: {mean_l2_difference:.4f}")
print(f"Mean L2 norm of original features: {mean_original_norm:.4f}")
