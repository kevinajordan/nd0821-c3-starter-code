import pytest
import requests
from unittest.mock import patch

# Define the base URL of the live API server
BASE_URL = "http://localhost:8000"


def test_get_root():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {
        "greeting": "Welcome to the Census Data API!",
        "description": "This API predicts salary based on census data.",
        "endpoints": {
            "/": "This information",
            "/predict": "POST endpoint for salary prediction"
        }
    }


@pytest.mark.parametrize("mocked_prediction", [
    ([1]),
    ([0])
])
@patch("main.inference")
def test_post_predict(mock_inference, mocked_prediction):
    """
    Test the POST /predict endpoint with mocked inference results.

    This test function checks if the API endpoint returns a successful response
    and includes a probability value in the response.

    Args:
        mock_inference (MagicMock): Mocked inference function
        mocked_prediction (list):
            - Mocked prediction from the model (either [0] or [1])

    The test sends a POST request with sample census data and asserts:
    1. The response status code is 200 (OK)
    2. The response includes a probability value of type float
    """
    mock_inference.return_value = mocked_prediction

    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = requests.post(f"{BASE_URL}/predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.json()["probability"], float)
