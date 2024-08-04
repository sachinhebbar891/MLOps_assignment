import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os

def train_model(run_name, C_value):
    # Set up MLflow experiment
    mlflow.set_experiment("Logistic_Regression_Experiment")
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Load data
        data = pd.read_csv('data/train.csv')
        X = data.drop('Outcome', axis=1)
        y = data['Outcome'].copy()

        # Initialize and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)

        # Initialize and fit model
        model = LogisticRegression(C=C_value)
        model.fit(X_train_scaled, y)

        # Make predictions and evaluate
        y_pred = model.predict(X_train_scaled)
        accuracy = accuracy_score(y, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("C_value", C_value)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact('data/train.csv')

        # Save scaler and model locally
        scaler_path = 'model/standard_scaler_mlflow.joblib'
        model_path = 'model/lr_model_mlflow.joblib'
        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)

        # Log the saved joblib files as artifacts
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(model_path)

        # Clean up local files
        os.remove(scaler_path)
        os.remove(model_path)

# Run experiments with different parameters
if __name__ == "__main__":
    # Example runs with different parameters
    train_model("Run_with_C_0.1", C_value=0.1)
    train_model("Run_with_C_1.0", C_value=1.0)
    train_model("Run_with_C_10.0", C_value=10.0)
