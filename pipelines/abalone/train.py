"""Trains an XGBoost model using provided hyperparameters."""

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "mlflow==2.13.2",
    "sagemaker-mlflow==0.1.0",
])

import argparse
import logging
import os
import pandas as pd
import pickle as pkl
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting training script.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lambda", type=float, dest="lambda_param")
    parser.add_argument("--num_round", type=int, required=True, help="Number of boosting rounds")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum tree depth")
    parser.add_argument("--eta", type=float, required=True, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=True, help="Minimum loss reduction")
    parser.add_argument("--min_child_weight", type=float, required=True, help="Minimum sum of child weights")
    parser.add_argument("--subsample", type=float, required=True, help="Subsample ratio")
    parser.add_argument("--objective", type=str, required=True, help="Objective function")

    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(os.path.join("/opt/ml/input/data/train/train.csv"), header=None)
    validation_df = pd.read_csv(os.path.join("/opt/ml/input/data/validation/validation.csv"), header=None)

    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_validation = validation_df.iloc[:, 0]
    X_validation = validation_df.iloc[:, 1:]

    logger.info("Data loaded. Starting training.")
    
    # Configure MLflow
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    tracking_server_arn = os.getenv("MLFLOW_TRACKING_SERVER_ARN")
    run_id = os.getenv("MLFLOW_RUN_ID")

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)    
    
    with mlflow.start_run():
        with mlflow.start_run(run_name=f"ModelTraining", nested=True) as training_run:
            training_run_id = training_run.info.run_id
            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
                model_format="xgb",
            )

            # Train the model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalidation = xgb.DMatrix(X_validation, label=y_validation)

            params = {
                "objective": args.objective,
                "max_depth": args.max_depth,
                "eta": args.eta,
                "gamma": args.gamma,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "reg_lambda": args.lambda_param,
                "reg_alpha": args.alpha,
                "eval_metric": "rmse",
            }
            evals = [(dtrain, "train"), (dvalidation, "validation")]

            model = xgb.train(params, dtrain, num_boost_round=args.num_round, evals=evals)

            # Save the model
            
            model_location = "/opt/ml/model" + "/xgboost-model"
            pkl.dump(model, open(model_location, "wb"))
            logging.info("Stored trained model at {}".format(model_location))

            logger.info("Model training complete.")
