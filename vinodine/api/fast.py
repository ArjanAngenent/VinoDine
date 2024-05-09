import pandas as pd
# $WIPE_BEGIN
from vinodine.ml_logic.registry import load_model
from vinodine.ml_logic.preprocessor import preprocess_features
# $WIPE_END
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

file_path = '~/code/ArjanAngenent/VinoDine/X_processed.pkl'
# Load the pickle file
with open(file_path, 'rb') as f:
    preprocessor = pickle.load(f)

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# $WIPE_BEGIN
# :bulb: Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = load_model()
# $WIPE_END
# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        Type: str,
        ABV: float,
        Body: str,
        Acidity: str
    ):      # 1
    """
    Make a single food prediction.

    """
    # $CHA_BEGIN
    # :bulb: Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None
    #X_processed = preprocess_features(X_pred)
    y_pred = model.predict(preprocessor.transform(X_pred))
    # :warning: fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(fare=float(y_pred))
    # $CHA_END
@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
