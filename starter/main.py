from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging

from starter.ml.model import inference
from starter.ml.data import process_data

app = FastAPI()

# Load the model, encoder, and label binarizer
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")


class CensusData(BaseModel):
    """
    CensusData model for input validation.

    Attributes:
        age (int): Age of the individual.
        workclass (str): Work class of the individual.
        fnlgt (int): Final weight.
        education (str): Education level.
        education_num (int): Number of years of education.
        marital_status (str): Marital status.
        occupation (str): Occupation.
        relationship (str): Relationship status.
        race (str): Race of the individual.
        sex (str): Sex of the individual.
        capital_gain (int): Capital gain.
        capital_loss (int): Capital loss.
        hours_per_week (int): Hours worked per week.
        native_country (str): Native country.
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.get("/")
async def root():
    """
    Root endpoint for the Census Data API.

    This endpoint provides basic information about the API,
    including a greeting, a brief description of its purpose,
    a brief description of its purpose, and a list of available endpoints.

    Returns:
        dict: A dictionary containing the API information.
            - greeting (str): A welcome message.
            - description (str): 
            A brief description of the API's purpose.
            - endpoints (dict): 
                A dictionary of available endpoints and their descriptions.
    """
    return {
        "greeting": "Welcome to the Census Data API!",
        "description": "This API predicts salary based on census data.",
        "endpoints": {
            "/": "This information",
            "/predict": "POST endpoint for salary prediction"
        }
    }

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.post("/predict")
async def predict(data: CensusData):
    """
    Predict salary based on census data.

    This endpoint receives census data as input 
    and returns a salary prediction.

    Args:
        data (CensusData): The input census data.

    Returns:
        dict: 
        A dictionary containing the prediction label and probability.
            - prediction (str): 
                The predicted salary category (">50K" or "<=50K").
            - probability (float): 
                The probability of the prediction.

    Raises:
        HTTPException: If an error occurs during the prediction process.
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict(by_alias=True)])
        logging.info("Input data: %s", input_data)

        # Process the input data
        cat_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]
        X, _, _, _ = process_data(
            input_data, categorical_features=cat_features, 
            training=False, encoder=encoder, lb=lb, label=None
        )
        logging.info("Processed data: %s", X)

        # Make prediction
        prediction = inference(model, X)
        logging.info("Raw prediction: %s", prediction)

        # Convert prediction back to label
        prediction_label = lb.inverse_transform(prediction)[0]
        logging.info("Prediction label: %s", prediction_label)

        return {
            "prediction": prediction_label,
            "probability": float(prediction[0])
        }
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
