from unittest import mock

import numpy as np
import pandas as pd

from FSDS_.score import score_model


def test_score_model():
    data = {
        "longitude": [1, 2],
        "latitude": [3, 4],
        "housing_median_age": [10, 20],
        "total_rooms": [20, 40],
        "total_bedrooms": [10, 20],
        "population": [100, 400],
        "households": [2, 4],
        "median_income": [10000, 30000],
        "median_house_value": [5000, 9000],
        "ocean_proximity": ["NEAR BAY", "INLAND"],
    }

    df = pd.DataFrame(data)
    fake_imputer = mock.MagicMock()
    fake_imputer.transform.return_value = df.drop(
        columns=["ocean_proximity", "median_house_value"]
    ).values

    with mock.patch("FSDS_.score.preprocess_data") as mock_preprocess:
        df_processed = {
            "longitude": [1, 2],
            "latitude": [3, 4],
            "housing_median_age": [10, 20],
            "total_rooms": [20, 40],
            "total_bedrooms": [10, 20],
            "population": [100, 400],
            "households": [2, 4],
            "median_income": [10000, 30000],
            "rooms_per_household": [10.0, 10.0],
            "bedrooms_per_room": [0.5, 0.5],
            "population_per_household": [50.0, 100.0],
            "median_house_value": [5000, 9000],
            "INLAND": [0, 1],
        }
        mock_preprocess.return_value = (df_processed, fake_imputer)

        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([4500, 9300])

        rmse = score_model(mock_model, df, fake_imputer)

        expected_output = np.sqrt(np.mean([(4500 - 5000) ** 2, (9300 - 9000) ** 2]))
        assert abs(rmse - expected_output) < 1e-8
