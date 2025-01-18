import tensorflow as tf  # type: ignore
from model import create_cnn
from data_processing import preprocess_data_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json


def train_model():
    print('Starting training')

    X_train, X_test, y_train, y_test = preprocess_data_cnn('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\dataset10k\\0', 'C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\dataset10k\\1')

    model = create_cnn(input_shape=(224, 224, 3))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,  
        validation_data=(X_test, y_test),              
        epochs=10,
        verbose=1
    )

    print('Training finished')
    
    model.save('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\deepfake_cnn_model_10k.keras')
    with open('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\\training_history_10k.json', 'w') as f:
        json.dump(history.history, f)

    print('Model saved')


if __name__ == "__main__":
    train_model()