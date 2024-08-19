import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
import xgboost as xgb
import joblib
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, encoder, lb, model_path):
    """
    Save the model, encoder, and label binarizer(lb) to the model_path.
    """
    joblib.dump(model, model_path)
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(lb, "model/lb.pkl")

def compute_sliced_metrics(model, X, y, feature_index):
    """
    Compute performance metrics on slices of the data for a given feature.

    Inputs
    ------
    model : 
        Trained machine learning model.
    X : np.array
        Encoded feature data.
    y : np.array
        True labels.
    feature_index : int
        Index of the feature to slice on.

    Returns
    -------
    slice_metrics : dict
        Dictionary with feature values as keys and their corresponding metrics as values.
    """
    slice_metrics = {}
    unique_values = np.unique(X[:, feature_index])

    for value in unique_values:
        mask = X[:, feature_index] == value
        X_slice = X[mask]
        y_slice = y[mask]

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        slice_metrics[value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "support": mask.sum()
        }

    return slice_metrics

def print_slice_metrics(slice_metrics, feature_name):
    """
    Print the metrics for each slice of the specified feature.

    Inputs
    ------
    slice_metrics : dict
        Dictionary with feature values as keys and their corresponding metrics as values.
    feature_name : str
        Name of the feature that was sliced on.
    """
    print(f"Slice metrics for feature: {feature_name}")
    print("-" * 50)
    for value, metrics in slice_metrics.items():
        print(f"Value: {value}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F-beta: {metrics['fbeta']:.4f}")
        print(f"  Support: {metrics['support']}")
        print("-" * 50)