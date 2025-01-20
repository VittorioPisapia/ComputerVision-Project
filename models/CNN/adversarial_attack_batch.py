import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from art.attacks.evasion import FastGradientMethod, CarliniL2Method # type: ignore
from art.estimators.classification import KerasClassifier # type: ignore
from data_processing import preprocess_data_cnn
from model import create_cnn

tf.compat.v1.disable_eager_execution()

input_shape = (224, 224, 3)
real_dir = 'C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\new_dataset\Training_1500\\0'
fake_dir = 'C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\new_dataset\Training_1500\\1'

X_train, X_test, y_train, y_test = preprocess_data_cnn(real_dir, fake_dir)

model = create_cnn(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,              
    epochs=10,
    batch_size=32,
    verbose=1
)

batch_size = 100
X_batch = X_test[:batch_size]
y_batch = y_test[:batch_size]

benign_preds = model.predict(X_batch)
benign_accuracy = np.mean((benign_preds > 0.5).astype(int).flatten() == y_batch)
print(f"Accuracy on benign test examples: {benign_accuracy * 100:.2f}%")

classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

attack = FastGradientMethod(estimator=classifier, eps=0.08)
x_batch_adv = attack.generate(x=X_batch)

adv_preds = model.predict(x_batch_adv)
adv_accuracy = np.mean((adv_preds > 0.5).astype(int).flatten() == y_batch)
print(f"Accuracy on adversarial test examples: {adv_accuracy * 100:.2f}%")

original_preds = (model.predict(X_batch) > 0.5).astype(int).flatten()

adversarial_preds = (model.predict(x_batch_adv) > 0.5).astype(int).flatten()

successful_attacks = np.sum(original_preds != adversarial_preds)
total_samples = len(X_batch)

print(f"Number of successful adversarial attacks: {successful_attacks} out of {batch_size}")
print(f"Success rate of adversarial attacks: {successful_attacks / total_samples * 100:.2f}%")
print(model.predict(X_batch)[:20])
print(model.predict(x_batch_adv)[:20])

