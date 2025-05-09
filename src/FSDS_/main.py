from FSDS_.ingest import fetch_housing_data, load_housing_data
from FSDS_.score import preprocess_data, score_model
from FSDS_.train import stratified_split, linear_reg, random_forest


def main():
    print("Loading data...")
    fetch_housing_data()
    housing = load_housing_data()

    print("Splitting data with stratified sampling...")
    train_set, test_set = stratified_split(housing)

    print("Preprocess training data...")
    X_train, imputer = preprocess_data(train_set.drop("median_house_value", axis=1))
    y_train = train_set["median_house_value"]

    print("Training Linear Regression...")
    lin_model = linear_reg(X_train, y_train)

    print("Training Random Forest with Grid Search...")
    rf_model, _ = random_forest(X_train, y_train)

    print("Scoring Linear Regression...")
    lin_rmse = score_model(lin_model, test_set, imputer)

    print("Scoring Random Forest...")
    rf_rmse = score_model(rf_model, test_set, imputer)

    print(f"Linear Regression RMSE: {lin_rmse:.2f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")


if __name__ == "__main__":
    main()
