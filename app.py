from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load dataset
df = pd.read_excel("Pump-time-series-data-V1.xlsx", engine="openpyxl")

# Define features
features = ["vibration", "temperature", "power_consumption"]

# Initialize FastAPI app
app = FastAPI()

# Train classification model
def train_classification_model(days_threshold):
    df["label_within_days"] = (df["Time to failure (Days)"] <= days_threshold).astype(int)
    X = df[features]
    y = df["label_within_days"]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf

# Train regression model
def train_regression_model():
    X = df[features]
    y = df["Time to failure (Days)"]
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X, y)
    return reg

# Load models
clf_model = train_classification_model(days_threshold=1)
reg_model = train_regression_model()

# Request models
class ClassificationRequest(BaseModel):
    days_threshold: int

class RegressionRequest(BaseModel):
    pump_tag: str

@app.post("/predict-failures")
def predict_failures(request: ClassificationRequest):
    days_threshold = request.days_threshold
    model = train_classification_model(days_threshold)
    df["label_within_days"] = (df["Time to failure (Days)"] <= days_threshold).astype(int)
    X = df[features]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    df["predicted_failure"] = predictions
    df["failure_probability"] = probabilities
    failing_pumps = df[df["predicted_failure"] == 1][["Pump Tag", "failure_probability", "Time to failure (Days)"]]
    if failing_pumps.empty:
        return {"message": f"No pumps predicted to fail within {days_threshold} days."}
    grouped = failing_pumps.groupby("Pump Tag").agg(
        avg_failure_probability=("failure_probability", "mean"),
        avg_days_remaining=("Time to failure (Days)", "mean")
    ).reset_index()
    filtered = grouped[grouped["avg_days_remaining"] <= days_threshold]
    if filtered.empty:
        return {"message": f"No pumps predicted to fail within {days_threshold} days (after filtering by average days remaining)."}
    return filtered.to_dict(orient="records")

@app.post("/predict-time-to-failure")
def predict_time_to_failure(request: RegressionRequest):
    pump_tag = request.pump_tag
    pump_data = df[df["Pump Tag"] == pump_tag]
    if pump_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for pump tag: {pump_tag}")
    X = pump_data[features]
    prediction = reg_model.predict(X)
    avg_days = prediction.mean()
    max_days = df["Time to failure (Days)"].max()
    min_days = df["Time to failure (Days)"].min()
    if max_days == min_days:
        probability = 1.0 if avg_days <= max_days else 0.0
    else:
        probability = 1 - (avg_days - min_days) / (max_days - min_days)
    probability = np.clip(probability, 0, 1)
    return {
        "pump_tag": pump_tag,
        "predicted_days_to_failure": round(avg_days, 2),
        "estimated_failure_probability": round(probability, 2)
    }
