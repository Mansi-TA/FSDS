import json
import os
import sys

import pandas as pd
from evidently.metric_preset import RegressionPreset
from evidently.report import Report


def load_data(reference_path, current_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    return reference, current


def generate_performance_report(reference_df, current_df, report_dir="monitoring"):
    os.makedirs(report_dir, exist_ok=True)

    # Creating report
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    # Save HTML
    html_path = os.path.join(report_dir, "report.html")
    report.save_html(html_path)
    print(f"HTML Report saved to {html_path}")

    # Save json for drift
    json_path = os.path.join(report_dir, "report.json")
    report.save_json(json_path)
    print(f"JSON Report saved to {json_path}")

    return json_path


def check_performance_drift(json_path):
    with open(json_path, "r") as f:
        report_data = json.load(f)

    drift_detected = False
    for metric in report_data.get("metrics", []):
        if metric.get("metric") == "RegressionQualityMetric":
            result = metric.get("result", {})

            # Extract RMSE values
            current_rmse = result.get("current", {}).get("rmse")
            reference_rmse = result.get("reference", {}).get("rmse")

            print(f"Reference RMSE: {reference_rmse}")
            print(f"Current RMSE: {current_rmse}")

            if current_rmse is not None and reference_rmse is not None:
                rmse_increase = (current_rmse - reference_rmse) / reference_rmse
                print(f"RMSE increase: {rmse_increase:.2%}")
                drift_detected = bool(rmse_increase > 0.25)
                if rmse_increase > 0.25:
                    print("Performance drift detected")

                else:
                    print("No significant performance drift detected")

    with open("drift.txt", "w") as f:
        f.write(f"drift:{'true' if drift_detected else 'false'}")


if __name__ == "__main__":
    reference_path = "outputs/data/reference_data.csv"
    current_path = "outputs/data/current_data.csv"
    reference_df, current_df = load_data(reference_path, current_path)
    json_path = generate_performance_report(reference_df, current_df)
    check_performance_drift(json_path)
