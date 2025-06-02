import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import RegressionPreset
import json
import sys


def load_data(reference_path, current_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    return reference, current

def generate_performance_report(reference_df, current_df, report_dir="monitoring"):
    os.makedirs(report_dir,exist_ok=True)

    # Creating report
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_df,current_data=current_df)

    # Save HTML
    html_path = os.path.join(report_dir,"report.html")
    report.save_html(html_path)
    print(f"HTML Report saved to {html_path}")

    # Save json for drift
    json_path = os.path.join(report_dir,"report.json")
    report.save_json(json_path)
    print(f"JSON Report saved to {json_path}")

    return json_path

def check_performance_drift(json_path):
    with open(json_path,"r") as f:
        report_data = json.load(f)
    
    for metric in report_data.get("metrics",[]):
        if "name" in metric and metric["name"] == "RegressionPreset":
            result = metric.get("result", {})
            for target_name, values in result.items():
                # Checking for drift or issues
                if values.get("current", {}).get("dataset_drift") is True:
                    print("Performance drift detected.")
                    sys.exit(1)
    
    print("No performance drift detected.")
    sys.exit(0)
if __name__=="__main__":
    reference_path = "outputs/data/reference_data.csv"
    current_path = "outputs/data/current_data.csv"
    reference_df, current_df = load_data(reference_path,current_path)
    json_path = generate_performance_report(reference_df, current_df)
    check_performance_drift(json_path)

