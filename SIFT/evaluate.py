import pickle
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from data_preprocessing import preprocess_data_with_sift
import matplotlib.pyplot as plt # type: ignore

def evaluate_model_with_sift():
    print("Loading the model...")

    with open('D:\CV\Models_and_scalers\sift_svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('D:\CV\Models_and_scalers\scaler_SIFT.pkl', 'rb') as f:
        scaler = pickle.load(f)

    _, X_test, _, y_test = preprocess_data_with_sift('D:/CV/dataset/0', 'D:/CV/dataset/1')

    X_test = scaler.transform(X_test)

    print("Evaluating the SVM...")
    y_pred = svm.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    plt.show()

if __name__ == "__main__":
    evaluate_model_with_sift()
