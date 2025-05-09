from sklearn.metrics import mean_squared_error
from FSDS_.train import preprocess_data

def score_model(model, test_set, imputer):
    X_test = test_set.drop("median_house_value",axis = 1)
    y_test = test_set['median_house_value'].copy()

    X_test_prep, _ = preprocess_data(X_test,imputer,fit = False)
    predict = model.predict(X_test_prep)
    mse = mean_squared_error(y_test,predict)
    rmse = mse ** 0.5
    return rmse
