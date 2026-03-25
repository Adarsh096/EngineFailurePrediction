import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import optuna

# --- CONFIG ---
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
MLFLOW_TRACKING_URL = os.getenv('MLFLOW_TRACKING_URL')

# MLFLow Tracking Url From DagsHub
tracking_uri = MLFLOW_TRACKING_URL

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("engine_failure_project-training-experiment")

HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')
api = HfApi()

repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}"
Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv")
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv")

# XGBoost expects 1D arrays for labels, not DataFrames
ytrain_1d = ytrain.values.ravel()
ytest_1d = ytest.values.ravel()

# Splitting numeric list into two lists, one which are to be scale and another not to be scaled as they are categorical/ordinal values.
# Columns that need Scaling (Continuous)
numeric_scaling = [
  'engine_rpm',
  'lub_oil_pressure',
  'fuel_pressure',
  'coolant_pressure',
  'lub_oil_temp',
  'coolant_temp'
 ]

# Columns to keep as-is (Numerical but categorical/binary in nature)
numeric_passthrough = []

# categorical columns:
categorical_features = []

# Updating the ColumnTransformer
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_scaling),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("passthrough", numeric_passthrough) # Keep these exactly as they are.
)

# Define objective function
def objective(trial):
    # Suggest hyperparameters for XGBoost
    n_estimators = trial.suggest_int('n_estimators', 100, 105)
    max_depth = trial.suggest_int('max_depth', 8, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.03, log=True)
    subsample = trial.suggest_float('subsample', 0.8, 0.9)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.8, 0.9)
    gamma = trial.suggest_float('gamma', 0.3, 0.4)

    # Train model (using XGBoostClassifier)
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=1.6e-08, # L1 Regularization
        reg_lambda=2.78e-08, # L2 Regularization
        random_state=42,
        use_label_encoder=False, # Suppress warning
        eval_metric='logloss' # Suppress warning
    )

    # Creating pipeline
    model_pipeline = make_pipeline(preprocessor, clf)

    # Evaluate with cross-validation
    # Passing ytrain_1d to avoid shape warnings
    score = cross_val_score(model_pipeline, Xtrain, ytrain_1d, n_jobs=-1, cv=5, scoring='recall').mean()
    return score


# Starting MLFlow tracking
with mlflow.start_run(run_name="GHA_Automated_Training"):
    # adding metadata to know it is from our pipelines:
    mlflow.log_param("source", "github_actions")

    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # Extract the best hyperparameters from the study
    best_params = study.best_params
    # Logging best parameters
    mlflow.log_params(best_params)

    # Re-instantiate the "best" model
    # We include the fixed parameters (random_state, etc.) along with the tuned ones
    final_clf = XGBClassifier(
        **best_params,
        reg_alpha=1.6e-08, # L1 Regularization
        reg_lambda=2.78e-08, # L2 Regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Creating a final pipeline that includes the preprocessor
    # This ensures Xtest is correctly scaled/transformed before prediction
    best_pipeline = make_pipeline(preprocessor, final_clf)

    # Final fit on the full training data
    best_pipeline.fit(Xtrain, ytrain_1d)

    # Evaluation
    # Predict using the pipeline so Xtest gets scaled automatically
    y_pred = best_pipeline.predict(Xtest)
    test_report = classification_report(ytest_1d, y_pred, output_dict=True)

    # Safely accessing the '1' class metrics (binary classification 0/1)
    target_class = '1' if '1' in test_report else str(list(test_report.keys())[0])

    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_f1-score": test_report[target_class]['f1-score'],
        "test_precision-score": test_report[target_class]['precision'],
        "test_recall-score": test_report[target_class]['recall']
    })

    # Save and Log Artifact in MLFlow
    model_path = "best_model_v1.joblib"
    # Saving the entire pipeline so preprocessor is included in the file
    joblib.dump(best_pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Upload to HF Model Hub
    model_repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_MODEL_NAME}"
    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path, #src name
        path_in_repo="model.joblib", #target name
        repo_id=model_repo_id,
        repo_type="model",
    )
