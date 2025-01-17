import tensorflow as tf # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
from data_processing import preprocess_data_cnn
import json

def evaluate_model():
    model = tf.keras.models.load_model('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\deepfake_cnn_model_NODATA.keras')

    X_train, X_test, y_train, y_test = preprocess_data_cnn('D:\CV\\altro_dataset\Testing_500\\0', 'D:\CV\\altro_dataset\Testing_500\\1')

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    with open('D:\CV\Models_and_scalers\\training_history_NODATA.json', 'r') as f:
        loaded_history = json.load(f)
    
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()

    
    plt.plot(loaded_history['loss'], label='Train Loss')
    plt.plot(loaded_history['val_loss'], label='Validation Loss')
    plt.title('Loss durante il Training')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(loaded_history['accuracy'], label='Train Accuracy')
    plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuratezza durante il Training')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model()