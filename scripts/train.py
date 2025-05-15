import argparse
import logging
import os
import pickle

import pandas as pd

from FSDS_.train import linear_reg, preprocess_data, random_forest


def setup_logging(log_filename, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def main(input_folder, output_folder):
    setup_logging("train.log")
    logging.info("Starting training script...")

    os.makedirs(output_folder, exist_ok=True)

    train_path = os.path.join(input_folder, "train.csv")
    logging.info(f"Loading data from {train_path}...")
    train_set = pd.read_csv(train_path)

    X_train = train_set.drop(columns=["median_house_value"])
    y_train = train_set["median_house_value"]

    X_train_pre, imputer = preprocess_data(X_train)

    # Training linear regression model
    logging.info("Training Linear Regression model...")
    linear_model = linear_reg(X_train_pre, y_train)

    # Train Random Forest model
    logging.info("Training Random Forest model...")
    rf_model, _ = random_forest(X_train_pre, y_train)

    lin_model_path = os.path.join(output_folder, "linear_regression_model.pkl")
    rf_model_path = os.path.join(output_folder, "random_forest_model.pkl")

    logging.info(f"Saving Linear Regression model to {lin_model_path}...")
    with open(lin_model_path, "wb") as f:
        pickle.dump(linear_model, f)

    logging.info(f"Saving Random Forest model to {rf_model_path}...")
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf_model, f)

    logging.info("Model training and saving complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models and save as pickle files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the trained models as pickle files",
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing the training dataset",
    )
    args = parser.parse_args()
    main(input_folder=args.input_folder, output_folder=args.output_folder)
