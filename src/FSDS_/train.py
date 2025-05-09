import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
)
from FSDS_.feature import add_extra_features

def stratified_split(housing):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    strat_train_set = strat_train_set.drop("income_cat", axis=1)
    strat_test_set = strat_test_set.drop("income_cat", axis=1)

    return strat_train_set, strat_test_set

def preprocess_data(df,imputer = None,fit = True):
    df_num = df.drop("ocean_proximity", axis=1)
    if fit:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df_num)
    
    X = imputer.transform(df_num)

    df_prep = pd.DataFrame(X, columns=df_num.columns, index=df.index)
    df_prep = add_extra_features(df_prep)
    
    df_cat = pd.get_dummies(df[['ocean_proximity']],drop_first=True)
    df_prep = df_prep.join(df_cat)
    return df_prep, imputer

#Using linear regression
def linear_reg(X,y):
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    return lin_reg

# Using random forest
def random_forest(X,y):
    param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
        
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search