from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from data_preprocessing import preprocess_data_with_hog # type: ignore
import pickle
from sklearn.preprocessing import StandardScaler # type: ignore

def train_model():
    print("Starting training with HOG + SVM...")

    X_train, X_test, y_train, y_test = preprocess_data_with_hog('D:/CV/dataset/0', 'D:/CV/dataset/1')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='linear', probability=True, random_state=42)

    print("Training the SVM...")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Model saved as hog_svm_model.pkl, scaler saved as scaler_HOG.pkl
    with open('D:\CV\Models_and_scalers\hog_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('D:\CV\Models_and_scalers\scaler_HOG.pkl', 'wb') as f:
        pickle.dump(scaler, f)    
    
    print("Model saved as hog_svm_model.pkl in Models_and_scalers")

if __name__ == "__main__":
    train_model()
