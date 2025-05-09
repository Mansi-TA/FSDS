import numpy as np
import pandas as pd
from FSDS_.train import stratified_split,linear_reg

def test_stratified_split():
    data = {
        "median_income": [0.5, 1.0, 2.0, 3.5, 5.0, 7.0] * 10,
        "median_house_value": [100000] * 60,
        "ocean_proximity": ["INLAND"] * 60,
        "longitude": [1] * 60,
        "latitude": [1] * 60,
        "housing_median_age": [10] * 60,
        "total_rooms": [100] * 60,
        "total_bedrooms": [50] * 60,
        "population": [300] * 60,
        "households": [2] * 60,
    }
    df = pd.DataFrame(data)
    train_set, test_set = stratified_split(df) 
    
    assert len(train_set) + len(test_set) == len(df)
    assert abs(len(test_set)/len(df) - 0.2) < 0.05

def test_linear_reg():
    X = {
        "col_1":[1,2,3,4,5],
        "col_2":[5,6,7,8,9]
    }
    X = pd.DataFrame(X)
    y = pd.Series([100,200,300,400,500])

    model = linear_reg(X,y)
    predicted_value = model.predict(X)
    assert len(predicted_value) == len(y)
    assert isinstance(predicted_value,np.ndarray)