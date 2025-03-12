"""Evaluation script for measuring root mean squared error (RMSE) with MLflow."""

import sys
import subprocess
import os
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error

subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "mlflow==2.13.2",
    "sagemaker-mlflow",
])

import mlflow

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    
    # Configure MLflow
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    tracking_server_arn = os.getenv("MLFLOW_TRACKING_SERVER_ARN")
    run_id = os.getenv("MLFLOW_RUN_ID")

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        with mlflow.start_run(run_name="ModelEvaluation", nested=True):
            # Extract the model
            with tarfile.open(model_path) as tar:
                tar.extractall(path="")

            logger.debug("Loading xgboost model.")
            model = pickle.load(open("xgboost-model", "rb"))

            logger.debug("Reading test data.")
            test_path = "/opt/ml/processing/test/test.csv"
            df = pd.read_csv(test_path, header=None)
            y_test = df.iloc[:, 0]
            X_test = df.iloc[:, 1:]
            

            logger.info("Performing predictions against test data.")
            predictions = model.predict(xgboost.DMatrix(X_test))

            logger.debug("Calculating root mean squared error (RMSE).")
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            std = np.std(y_test - predictions)

            # Log metrics to MLflow
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("rmse_std", std)

            # Prepare evaluation report
            report_dict = {
                "regression_metrics": {
                    "rmse": {
                        "value": rmse,
                        "standard_deviation": std
                    },
                },
            }

            output_dir = "/opt/ml/processing/evaluation"
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            logger.info("Writing out evaluation report with RMSE: %f", rmse)
            evaluation_path = f"{output_dir}/evaluation.json"
            with open(evaluation_path, "w") as f:
                f.write(json.dumps(report_dict))
