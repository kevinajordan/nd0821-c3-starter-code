import pytest
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import xgboost as xgb
import joblib
import pandas as pd
from tempfile import TemporaryDirectory
from starter.ml.model import train_model, compute_model_metrics, inference, save_model
from starter.ml.data import process_data

@pytest.fixture(scope="module")
def synthetic_census_data():
    """
    Fixture to create synthetic census data for testing purposes.
    
    Returns:
        pd.DataFrame: A DataFrame containing synthetic census data.
    """
    data = pd.DataFrame({
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'fnlgt': [77516, 83311, 215646, 234721, 338409],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'education-num': [13, 13, 9, 7, 13],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'capital-gain': [2174, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40],
        'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba'],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
    })
    return data

def test_train_model(synthetic_census_data):
    """
    Test the train_model function using synthetic census data.
    
    Ensures that the function returns an XGBClassifier instance with the correct number of features.
    """
    print("synthetic_census_data:  ", synthetic_census_data)
    X, y, _, _ = process_data(
        synthetic_census_data,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )
    model = train_model(X, y)
    assert isinstance(model, xgb.XGBClassifier)
    assert model.n_features_in_ == X.shape[1]

def test_compute_model_metrics(synthetic_census_data):
    """
    Test the compute_model_metrics function using synthetic census data.
    
    Verifies that the function returns precision, recall, and fbeta scores between 0 and 1.
    """
    X, y, _, _ = process_data(
        synthetic_census_data,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = model.predict(X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference(synthetic_census_data):
    """
    Test the inference function using synthetic census data.
    
    Ensures that the function returns predictions as a numpy array with the correct shape.
    """
    X, y, _, _ = process_data(
        synthetic_census_data,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(synthetic_census_data),)


def test_process_data(synthetic_census_data):
    """
    Test the process_data function using synthetic census data.
    
    Checks the function's behavior in training mode, inference mode, and without a label.
    Ensures correct output types and shapes for each scenario.
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    # Test training mode
    X_train, y_train, encoder_train, lb_train = process_data(
        synthetic_census_data, 
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder_train, OneHotEncoder)
    assert isinstance(lb_train, LabelBinarizer)
    assert X_train.shape[0] == len(synthetic_census_data)
    assert y_train.shape == (len(synthetic_census_data),)

    # Test inference mode
    X_test, y_test, encoder_test, lb_test = process_data(
        synthetic_census_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder_train,
        lb=lb_train
    )

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert encoder_test is encoder_train
    assert lb_test is lb_train
    assert X_test.shape[0] == len(synthetic_census_data)
    assert y_test.shape == (len(synthetic_census_data),)

    # Test without label
    with pytest.raises(ValueError):
        process_data(
            synthetic_census_data.drop("salary", axis=1),
            categorical_features=categorical_features,
            training=True
        )

    # Test with non-existent label
    with pytest.raises(ValueError):
        process_data(
            synthetic_census_data,
            categorical_features=categorical_features,
            label="non_existent_label",
            training=True
        )

def test_model_performance(synthetic_census_data):
    """
    Test the overall performance of the model on the entire dataset.
    
    Trains the model on the full dataset and evaluates its performance.
    Ensures that the model achieves a minimum level of performance.
    """
    X, y, _, _ = process_data(
        synthetic_census_data,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert precision >= 0.5
    assert recall >= 0.5
    assert fbeta >= 0.5