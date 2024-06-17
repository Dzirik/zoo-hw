import pandas as pd
import pickle
import sys
from datetime import datetime
from numpy import std, mean
from os.path import join
from sklearn.metrics import mean_squared_error


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    with open("model_w04.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet")

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_hat = model.predict(X_val) # df["duration"].to_numpy()

    print(f"Mean Predicted Duration for {year:04d}-{month:02d} is {mean(y_hat)}")
