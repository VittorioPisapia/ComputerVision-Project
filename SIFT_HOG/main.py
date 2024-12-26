import train
import evaluate

def main():
    # Addestramento del modello
    #train.train_model_with_sift_hog()
    
    # Valutazione del modello
    evaluate.evaluate_model_with_sift_hog()

if __name__ == "__main__":
    main()