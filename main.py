import argparse
import logging
import os

import mlflow

from scripts.ingest_data import run_ingestion
from scripts.score import run_scoring
from scripts.train import run_training

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
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def main(base_output_folder):
    setup_logging("main.log")

    data_folder = os.path.join(base_output_folder, "data")
    model_folder = os.path.join(base_output_folder, "model")
    metric_folder = os.path.join(base_output_folder, "metric")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(metric_folder, exist_ok=True)

    logging.info("Starting the MLflow pipeline")
    # Setting the experiment
    mlflow.set_experiment("House Price Prediction Pipeline")

    with mlflow.start_run(run_name="Full_Pipeline") as parent_run:
        logging.info(f"Parent run ID:{parent_run.info.run_id}")

        mlflow.log_param("pipeline_output_folder", base_output_folder)

        logging.info("Doing data ingestion...")
        run_ingestion(output_folder=data_folder)

        logging.info("Doing data training...")
        run_training(input_folder=data_folder, output_folder=model_folder)

        logging.info("Scoring the model...")
        run_scoring(
            model_folder=model_folder,
            dataset_folder=data_folder,
            output_folder=metric_folder,
        )

        logging.info("Pipeline execution completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full ML pipeline using MLflow.")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder to store train/test data and models",
    )
    args = parser.parse_args()
    main(args.output_folder)
