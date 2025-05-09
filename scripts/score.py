import argparse
import logging
import os
import pickle

import pandas as pd

from FSDS_.score import *
from FSDS_.train import preprocess_data


def setup_logging(log_filename, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def main(model_folder, dataset_folder, output_folder=None):
    setup_logging("score.log")
    logging.info("Starting model scoring...")

    test_path = os.path.join(dataset_folder, "test.csv")
    lin_model_path = os.path.join(model_folder, "linear_regression_model.pkl")
    rf_model_path = os.path.join(model_folder, "random_forest_model.pkl")

    logging.info(f"Loading test dataset from {test_path}...")
    test_set = pd.read_csv(test_path)

    logging.info("Splitting test data into features and labels...")
    x_test = test_set.drop(columns=["median_house_value"])
    y_test = test_set["median_house_value"]

    logging.info("Extracting imputer from training preprocessing...")
    _, imputer = preprocess_data(x_test, fit=True)

    # Load and score Linear Regression model
    logging.info(f"Loading Linear Regression model from {lin_model_path}...")
    with open(lin_model_path, "rb") as f:
        lin_model = pickle.load(f)
    lin_rmse = score_model(lin_model, test_set, imputer)

    # Load and score Linear Regression model
    logging.info(f"Loading Random Forest model from {rf_model_path}...")
    with open(rf_model_path, "rb") as f:
        rf_model = pickle.load(f)
    rf_rmse = score_model(rf_model, test_set, imputer)

    logging.info(f"Linear Regression RMSE: {lin_rmse:.2f}")
    logging.info(f"Random Forest RMSE: {rf_rmse:.2f}")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        result_path = os.path.join(output_folder, "model_scores.txt")
        with open(result_path, "w") as f:
            f.write(f"Linear Regression RMSE: {lin_rmse:.2f}\n")
            f.write(f"Random Forest RMSE: {rf_rmse:.2f}\n")
        logging.info(f"Scores written to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score trained models on test data.")
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Folder containing model pickle files",
    )
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Folder containing test.csv"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Optional folder to save RMSE scores to a file",
    )
    args = parser.parse_args()
    main(args.model_folder, args.dataset_folder, args.output_folder)
