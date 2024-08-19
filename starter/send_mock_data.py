import requests

# Define the API endpoint URL
API_URL = "https://census-income-predictor.onrender.com/predict"

# Mock data to send to the API
mock_data = {
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

def send_predict_request(data):
    """
    Send a POST request to the /predict endpoint with the given data.

    Args:
        data (dict): The data to send in the request body.

    Returns:
        tuple: A tuple containing the status code and the JSON response from the API.
    """
    response = requests.post(API_URL, json=data)
    return response.status_code, response.json()

# Send the request and print the response
print("Sending POST request to /predict endpoint...")
status_code, response_data = send_predict_request(mock_data)
print(f"Response received (Status Code: {status_code}):")
print(response_data)

# Check if the response contains a probability
if "probability" in response_data:
    print(f"Probability: {response_data['probability']}")
else:
    print("No probability found in the response.")