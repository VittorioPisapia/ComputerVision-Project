from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from data_preprocessing import preprocess_data_with_sift_hog
import pickle
from sklearn.preprocessing import StandardScaler # type: ignore

def train_model_with_sift_hog():
    print("Starting training with SIFT + HOG + SVM...")

    X_train, X_test, y_train, y_test = preprocess_data_with_sift_hog('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\new_dataset\\Training_1500\\0', 'C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\new_dataset\\Training_1500\\1')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='linear', probability=True, random_state=42,verbose=True)

    print("Training the SVM...")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    with open('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\sift_hog_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('C:\\Users\\Tommaso\\Documents\\GitHub\\ComputerVision-Project\\Models_and_scalers\scaler_SIFT_HOG.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model saved as sift_hog_svm_model.pkl")

if __name__ == "__main__":
    train_model_with_sift_hog()

