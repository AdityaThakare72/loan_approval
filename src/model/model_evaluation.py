import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load the model
def load_model(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {model_path}")
        raise e
    except Exception as e:
        logging.error(f"Error occurred while loading model: {e}")
        raise e

# Function to load test data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        raise e

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=' Approved')
        recall = recall_score(y_true, y_pred, pos_label=' Approved')
        logging.info("Metrics calculated successfully.")
        return accuracy, precision, recall
    except Exception as e:
        logging.error(f"Error occurred while calculating metrics: {e}")
        raise e

# Function to save metrics to a file
def save_metrics(metrics, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=5)
        logging.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving metrics: {e}")
        raise e

# Main function to execute the process
def main():
    try:
        # Load model
        rf = load_model('models/model.pkl')

        # Load test data
        test_data = load_data('./data/processed/test_processed.csv')

        # Prepare test data
        x_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        # Predict on test data
        y_pred = rf.predict(x_test)

        # Calculate metrics
        accuracy, precision, recall = calculate_metrics(y_test, y_pred)

        # Prepare metrics dictionary
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        # Save metrics to file
        save_metrics(metrics_dict, 'models/metrics.json')

    except Exception as e:
        logging.error(f"Process failed: {e}")

if __name__ == "__main__":
    main()
