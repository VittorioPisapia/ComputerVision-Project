import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from art.attacks.evasion import FastGradientMethod # type: ignore
from art.estimators.classification import KerasClassifier # type: ignore
from data_processing import preprocess_data_cnn
from model import create_cnn

tf.compat.v1.disable_eager_execution()

# Parametri
input_shape = (224, 224, 3)
real_dir = 'D:/CV/altro_dataset/Training_1500/0'
fake_dir = 'D:/CV/altro_dataset/Training_1500/1'

# Preprocessa i dati
X_train, X_test, y_train, y_test = preprocess_data_cnn(real_dir, fake_dir)

# Crea e addestra il modello
model = create_cnn(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento
history = model.fit(
    X_train, y_train,              
    epochs=5,
    batch_size=32,
    verbose=1
)

# Valutazione su esempi benigni
benign_preds = model.predict(X_test)
benign_accuracy = np.mean((benign_preds > 0.5).astype(int).flatten() == y_test)
print(f"Accuracy on benign test examples: {benign_accuracy * 100:.2f}%")

# Crea il wrapper KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

# Genera attacchi avversariali
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_test)

# Valutazione su esempi avversariali
adv_preds = model.predict(x_test_adv)
adv_accuracy = np.mean((adv_preds > 0.5).astype(int).flatten() == y_test)
print(f"Accuracy on adversarial test examples: {adv_accuracy * 100:.2f}%")

# Predizioni su esempi originali
original_preds = (model.predict(X_test) > 0.5).astype(int).flatten()

# Predizioni su esempi avversariali
adversarial_preds = (model.predict(x_test_adv) > 0.5).astype(int).flatten()

# Calcolo del numero di attacchi avversariali effettuati con successo
successful_attacks = np.sum(original_preds != adversarial_preds)
total_samples = len(X_test)

print(f"Number of successful adversarial attacks: {successful_attacks}")
print(f"Success rate of adversarial attacks: {successful_attacks / total_samples * 100:.2f}%")
print(model.predict(X_test)[:20])
print(model.predict(x_test_adv)[:20])

