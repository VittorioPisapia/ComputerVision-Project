import tensorflow as tf  # type: ignore
from model import create_cnn
from data_processing import preprocess_data_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json


def train_model():
    print('Starting training')

    X_train, X_test, y_train, y_test = preprocess_data_cnn('D:\CV\\altro_dataset\Training_1500\\0', 'D:\CV\\altro_dataset\Training_1500\\1')

    model = create_cnn(input_shape=(224, 224, 3))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,  
        validation_data=(X_test, y_test),              
        epochs=20,
        verbose=1
    )

    print('Training finished')
    
    model.save('D:\CV\Models_and_scalers\deepfake_cnn_model_NODATA.keras')
    with open('D:\CV\Models_and_scalers\\training_history_NODATA.json', 'w') as f:
        json.dump(history.history, f)

    print('Model saved')


if __name__ == "__main__":
    train_model()