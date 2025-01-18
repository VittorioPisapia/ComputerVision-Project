import train
import evaluate
import time

def main():
    start_time = time.time()

    train.train_model()
    end_time = time.time()

    
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    evaluate.evaluate_model()

if __name__ == "__main__":
    main()