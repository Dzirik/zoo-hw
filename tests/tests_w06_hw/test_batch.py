"""
Tests
"""
import pytest
from datetime import datetime
from pandas import DataFrame

from src.w06_hw.batch import prepare_data

CATEGORICAL = ['PULocationID', 'DOLocationID']


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


DF_CORRECT = DataFrame(
    data=[
        ("-1", "-1", dt(1, 1), dt(1, 10), 9.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0)
    ],
    columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', "duration"]
)


def test_batch() -> None:
    """
    Tests batch function.
    """
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = DataFrame(data, columns=columns)

    assert prepare_data(df, CATEGORICAL).equals(DF_CORRECT)
