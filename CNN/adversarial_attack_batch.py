from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method # type: ignore
from art.estimators.classification import KerasClassifier # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from data_processing import preprocess_data_cnn

tf.compat.v1.disable_eager_execution()

model = tf.keras.models.load_model('D:/CV/Models_and_scalers/deepfake_cnn_model_NODATA.keras')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(0, 1))

batch_size = 100
_, X_test, _, _ = preprocess_data_cnn('D:\CV\\altro_dataset\Testing_500\\0', 'D:\CV\\altro_dataset\Testing_500\\1')
X_test_subset = X_test[:batch_size]


attack = FastGradientMethod(estimator=classifier, eps=1000)
#attack = ProjectedGradientDescent(estimator=classifier, eps=0.5, max_iter=10)
#attack = CarliniL2Method(classifier=classifier, confidence=0.5, max_iter=20)

print("Generating adversarial examples...")
adversarial_examples = attack.generate(x=X_test_subset)

original_predictions = classifier.predict(X_test_subset)
adversarial_predictions = classifier.predict(adversarial_examples)

success_rate = np.mean(np.argmax(original_predictions, axis=1) != np.argmax(adversarial_predictions, axis=1))

perturbations = adversarial_examples - X_test_subset
l2_norms = np.linalg.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1)
print(f"Mean L2 norm of perturbations: {np.mean(l2_norms):.4f}")

print(f"Success rate of adversarial attack: {success_rate * 100:.2f}%")
print(f"Original Predictions: {original_predictions[:5]}")
print(f"Adversarial Predictions: {adversarial_predictions[:5]}")
perturbations = adversarial_examples - X_test_subset
print(f"Max perturbation: {np.max(perturbations):.4f}")
print(f"Min perturbation: {np.min(perturbations):.4f}")
print(f"Mean perturbation: {np.mean(perturbations):.4f}")