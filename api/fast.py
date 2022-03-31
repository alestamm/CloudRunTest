from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import pandas as pd
import joblib
from TaxiFareModel.params import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, passenger_count
            ):

    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    formatted_pickup_datetime = utc_pickup_datetime.strftime(
        "%Y-%m-%d %H:%M:%S UTC")

    d = {
        "key": f"{pickup_datetime}",
        "pickup_datetime": f"{formatted_pickup_datetime}",
        "pickup_longitude": float(f"{pickup_longitude}"),
        "pickup_latitude": float(f"{pickup_latitude}"),
        "dropoff_longitude": float(f"{dropoff_longitude}"),
        "dropoff_latitude": float(f"{dropoff_latitude}"),
        "passenger_count": int(f"{passenger_count}")
    }

    df = pd.DataFrame(data=d, index=[0])

    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    y_pred = pipeline.predict(df)

    return {"fare": y_pred[0]}
