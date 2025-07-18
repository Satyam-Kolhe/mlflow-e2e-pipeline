# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_prep import load_and_split_data
import os

# Set MLflow tracking URI (where your tracking server data will be stored)
# For local development, this defaults to 'mlruns/' directory in your project.
# You can set it to a database (e.g., "sqlite:///mlruns.db") for persistence.
# For simplicity, we'll use the default local directory for now.
# mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Define an experiment name
EXPERIMENT_NAME = "Breast Cancer Classification"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_log_model(model_name, model_class, params, X_train, X_test, y_train, y_test):
    """
    Trains a model, logs its parameters, metrics, and the model itself to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Initialize and train the model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        mlflow.log_metrics(metrics)
        print(f"Logged metrics for {model_name}: {metrics}")

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{model_name}-BreastCancer", 
            input_example=X_train.head(5) 
        )
        print(f"Logged model for {model_name}")
        
        with open("model_summary.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"Metrics: {metrics}\n")
        mlflow.log_artifact("model_summary.txt")
        print(f"Logged artifact: model_summary.txt")

        # Return the run ID for later reference (e.g., for model registration)
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Experiment 1: Logistic Regression
    lr_params = {"solver": "liblinear", "penalty": "l1", "C": 0.5, "random_state": 42}
    print("\n--- Running Logistic Regression Experiment ---")
    lr_run_id = train_and_log_model("Logistic Regression", LogisticRegression, lr_params, X_train, X_test, y_train, y_test)

    # Experiment 2: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    print("\n--- Running Random Forest Experiment ---")
    rf_run_id = train_and_log_model("Random Forest", RandomForestClassifier, rf_params, X_train, X_test, y_train, y_test)

    # Experiment 3: Gradient Boosting
    gb_params = {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5, "random_state": 42}
    print("\n--- Running Gradient Boosting Experiment ---")
    gb_run_id = train_and_log_model("Gradient Boosting", GradientBoostingClassifier, gb_params, X_train, X_test, y_train, y_test)

    # Experiment 4: Support Vector Machine (SVC)
    svc_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42}
    print("\n--- Running SVC Experiment ---")
    svc_run_id = train_and_log_model("SVC", SVC, svc_params, X_train, X_test, y_train, y_test)

    print("\nAll experiments completed. Launch MLflow UI to view results: `mlflow ui`")