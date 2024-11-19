from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Flask app setup
app = Flask(__name__)

# Prefect Tasks
@task
def fetch_data():
    """Fetch the Iris dataset."""
    data = load_iris(as_frame=True)
    df = data.frame
    target = data.target.name
    return df, target

@task
def preprocess_data(df, target):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@task
def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

@task
def deploy_model(model, accuracy, threshold=0.9):
    """Save the model if the accuracy is above the threshold."""
    if accuracy >= threshold:
        joblib.dump(model, "best_model.pkl")
        return "Model saved successfully!"
    else:
        return "Model did not meet the accuracy threshold."

# Prefect Flow
@flow
def ml_pipeline():
    """Orchestrate the entire ML pipeline."""
    df, target = fetch_data()
    X_train, X_test, y_train, y_test = preprocess_data(df, target)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    result = deploy_model(model, accuracy)
    return {"accuracy": accuracy, "result": result}

# Flask API endpoints
@app.route("/")
def home():
    return "Welcome to the ML Pipeline API! Use /run-pipeline to start the pipeline."

@app.route("/run-pipeline", methods=["GET"])
def run_pipeline():
    """Trigger the ML pipeline via an API call."""
    result = ml_pipeline()
    return jsonify(result)

@app.route("/model", methods=["GET"])
def check_model():
    """Check if the model exists."""
    try:
        model = joblib.load("best_model.pkl")
        return "Model exists and is ready to use."
    except FileNotFoundError:
        return "No model has been saved yet."

# Running Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
