# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, save_model, compute_model_metrics, inference, compute_sliced_metrics, print_slice_metrics
import pandas as pd
import joblib
import numpy as np
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def save_evaluations(overall_metrics, slice_metrics, log_path):
    """
    Save the overall metrics and slice metrics to a log file.
    
    Inputs:
    ------
    overall_metrics : dict
        Dictionary containing overall model metrics
    slice_metrics : dict
        Dictionary containing metrics for each slice of each feature
    log_path : str
        Path to save the log file
    """
    with open(log_path, 'w') as f:
        f.write("Overall Model Metrics:\n")
        f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
        f.write(f"Recall: {overall_metrics['recall']:.4f}\n")
        f.write(f"F-beta: {overall_metrics['fbeta']:.4f}\n\n")
        
        f.write("Slice Metrics:\n")
        for feature, metrics in slice_metrics.items():
            f.write(f"\nFeature: {feature}\n")
            for value, value_metrics in metrics.items():
                f.write(f"  Value: {value}\n")
                f.write(f"    Precision: {value_metrics['precision']:.4f}\n")
                f.write(f"    Recall: {value_metrics['recall']:.4f}\n")
                f.write(f"    F-beta: {value_metrics['fbeta']:.4f}\n")
                f.write(f"    Support: {value_metrics['support']}\n")

# Load and prepare data
logger.info("Loading data")
data = pd.read_csv("data/census.csv")

logger.info("Splitting data")
train, test = train_test_split(data, test_size=0.20)

logger.info("Processing data")
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logger.info("Processing test data")
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save model
logger.info("Training model")
model = train_model(X_train, y_train)

logger.info("Saving model")
save_model(model, encoder, lb, "model/model.pkl")

# Compute overall metrics
logger.info("Computing overall metrics")
y_pred = inference(model, X_test)
overall_metrics = dict(zip(["precision", "recall", "fbeta"], compute_model_metrics(y_test, y_pred)))

# Compute slice metrics
logger.info("Computing slice metrics")
slice_metrics = {}
for i, feature in enumerate(cat_features):
    slice_metrics[feature] = compute_sliced_metrics(model, X_test, y_test, i)
    print_slice_metrics(slice_metrics[feature], feature)

# Save evaluations
logger.info("Saving evaluations")
save_evaluations(overall_metrics, slice_metrics, "model/model_evaluation.log")

logger.info("Model training and evaluation completed")