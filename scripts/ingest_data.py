import argparse
import os
import pandas as pd
import logging

from FSDS_.ingest import *
from FSDS_.train import stratified_split

def setup_logging(log_filename, log_dir="logs"):
    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,log_filename)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)

def main(output_folder):
    setup_logging("ingest.log")
    logging.info("Starting data ingestion...")

    os.makedirs(output_folder, exist_ok=True)

    logging.info("Fetching house data...")
    fetch_housing_data()
    house_df = load_housing_data()

    logging.info("Performing stratified split...")
    train_set, test_set = stratified_split(house_df)

    train_path = os.path.join(output_folder,"train.csv") 
    test_path = os.path.join(output_folder,"test.csv")

    logging.info("Storing training and testing set...")
    train_set.to_csv(train_path, index= False)
    test_set.to_csv(test_path, index= False) 
    
    logging.info("Ingestion completed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download and split housing data")
    parser.add_argument("--output_folder", 
                        type = str,
                        required = True,
                        help="Path to train.csv and test.csv")
    args = parser.parse_args()
    main(args.output_folder)


          
