import argparse
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

from FSDS_.ingest import fetch_housing_data, load_housing_data
from FSDS_.train import stratified_split

mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()

# Check if experiment exists by name
experiment_name = "Default"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    client.create_experiment(experiment_name)

# Set experiment
mlflow.set_experiment(experiment_name)

# Ensure default experiment exists
if not mlflow.get_experiment_by_name("Default"):
    mlflow.create_experiment("Default")


def setup_logging(log_filename, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def run_ingestion(output_folder):

    setup_logging("ingest.log")
    logging.info("Starting data ingestion...")

    with mlflow.start_run(run_name="Data Ingestion", nested=True) as run:

        print(f"Ingestion Run id:{run.info.run_id}")
        mlflow.log_param("output_folder", output_folder)

        os.makedirs(output_folder, exist_ok=True)

        logging.info("Fetching house data...")
        fetch_housing_data()
        house_df = load_housing_data()

        logging.info("Performing stratified split...")
        train_set, test_set = stratified_split(house_df)

        train_path = os.path.join(output_folder, "train.csv")
        test_path = os.path.join(output_folder, "test.csv")

        logging.info("Storing training and testing set...")
        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        mlflow.log_artifact(train_path, artifact_path="ingested_data")
        mlflow.log_artifact(test_path, artifact_path="ingested_data")

        logging.info("Ingestion completed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download and split housing data")
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to train.csv and test.csv",
    )
    args = parser.parse_args()
    run_ingestion(args.output_folder)
