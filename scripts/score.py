import argparse
import logging
import os
import pickle

import mlflow
import pandas as pd

from FSDS_.score import score_model

mlflow.set_tracking_uri("file:./mlruns")

# Ensure default experiment exists
if not mlflow.get_experiment_by_name("Default"):
    mlflow.create_experiment("Default")


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


def run_scoring(model_folder, dataset_folder, output_folder=None):
    setup_logging("score.log")
    logging.info("Starting model scoring...")

    with mlflow.start_run(run_name="Model Scoring", nested=True) as run:
        print(f"Score run id:{run.info.run_id}")
        test_path = os.path.join(dataset_folder, "test.csv")
        lin_model_path = os.path.join(model_folder, "linear_regression_model.pkl")
        rf_model_path = os.path.join(model_folder, "random_forest_model.pkl")

        logging.info(f"Loading test dataset from {test_path}...")
        test_set = pd.read_csv(test_path)

        logging.info("Extracting imputer from training preprocessing...")
        imputer_path = os.path.join(model_folder, "imputer.pkl")
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)

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

        mlflow.log_metric("Linear_Regression_RMSE", lin_rmse)
        mlflow.log_metric("Random_Forest_RMSE", rf_rmse)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            result_path = os.path.join(output_folder, "model_scores.txt")
            with open(result_path, "w") as f:
                f.write(f"Linear Regression RMSE: {lin_rmse:.2f}\n")
                f.write(f"Random Forest RMSE: {rf_rmse:.2f}\n")
            logging.info(f"Scores written to {result_path}")
            mlflow.log_artifact(result_path, artifact_path="scoring_outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score trained models on test data.")
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Folder containing model pickle files",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Folder containing test.csv",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Optional folder to save RMSE scores to a file",
    )
    args = parser.parse_args()
    run_scoring(args.model_folder, args.dataset_folder, args.output_folder)
