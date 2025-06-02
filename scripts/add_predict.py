import pandas as pd
import numpy as np
import pickle
import argparse

from FSDS_.train import preprocess_data

def generate_predictions_drifted_data(test_data, imputer_path, model_path, output_path, noise_level):
    drifted_data = test_data.copy()
    drifted_data_X = drifted_data.drop('median_house_value',axis=1)
    drifted_data_y = drifted_data['median_house_value']

    # Load imputer
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f) 
    
    drifted_data_X_pre, _ = preprocess_data(drifted_data_X,imputer,fit=False)

    

    # Creating drifted data
    np.random.seed(seed=42)
    num_cols = drifted_data_X_pre.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = drifted_data_X_pre[col].std()
        noise = np.random.normal(0,noise_level*std,size=len(drifted_data_X_pre))
        drifted_data_X_pre[col] += noise
    
    # Calculating predictions with noise data
    with open(model_path,"rb") as f:
        model = pickle.load(f)
    
    y_pred =  model.predict(drifted_data_X_pre)

    # Combining into final df
    drifted_data = drifted_data_X_pre.copy()
    drifted_data['target'] = drifted_data_y
    drifted_data['prediction'] = y_pred

    # Saving drifted data as current
    drifted_data.to_csv(output_path,index=False)
    print(f"Current file saved at {output_path}")

def generate_predictions_actual_data(test_data, imputer_path, model_path, output_path):
    
    reference_X = test_data.drop(['median_house_value'],axis=1)

    # Load imputer
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f) 
    
    reference_X_pre, _ = preprocess_data(reference_X,imputer,fit=False)

    # Load model and predict
    with open(model_path,"rb") as f:
        model = pickle.load(f)

    test_data['prediction'] = model.predict(reference_X_pre)
    test_data.rename(columns={'median_house_value':'target'},inplace=True)
    test_data.to_csv(output_path,index=False)
    print(f"Reference file saved at {output_path}")

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",type=str,default="outputs/data/test.csv")
    parser.add_argument("--imputer_path",type=str,default="outputs/model/imputer.pkl")
    parser.add_argument("--model_path",type=str,default="outputs/model/random_forest_model.pkl")
    parser.add_argument("--reference_output_path",type=str,default="outputs/data/reference_data.csv")
    parser.add_argument("--current_output_path",type=str,default="outputs/data/current_data.csv")
    parser.add_argument("--noise_level",type=float,default=0.1)
    args = parser.parse_args()
    test_data = pd.read_csv(args.test_path)
    generate_predictions_actual_data(
        test_data.copy(),
        imputer_path=args.imputer_path,
        model_path=args.model_path,
        output_path=args.reference_output_path)
    
    generate_predictions_drifted_data(
        test_data,
        imputer_path=args.imputer_path,
        model_path=args.model_path,
        output_path=args.current_output_path,
        noise_level=args.noise_level)
    
