import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
# Load your dataset
df = pd.read_excel("Pump-time-series-data-V1.xlsx")

# Example: Select features and target
features = ["vibration", "temperature", "power_consumption"]
#df["label_within_days"] = 0  # placeholder for classification


# --- Classification Model ---
def train_classification_model(days_threshold):
    df["label_within_days"] = (df["Time to failure (Days)"] <= days_threshold).astype(int)
    X = df[features]
    y = df["label_within_days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print("Classification model trained.")
    return clf

# --- Example Usage ---
# Ask user for days threshold
try:
    days_threshold = int(input("Enter the number of days to check for likely failures: "))
except ValueError:
    print("Invalid input. Using default threshold of 10 days.")
    days_threshold = 1
clf_model = train_classification_model(days_threshold=days_threshold)

# Predict failing pumps and their probabilities
df["label_within_days"] = (df["Time to failure (Days)"] <= days_threshold).astype(int)
X = df[features]
predictions = clf_model.predict(X)
probabilities = clf_model.predict_proba(X)[:, 1]  # Probability of failure (class 1)

df["predicted_failure"] = predictions
df["failure_probability"] = probabilities

failing_pumps = df[df["predicted_failure"] == 1][["Pump Tag", "failure_probability", "Time to failure (Days)"]]

if not failing_pumps.empty:
    grouped = failing_pumps.groupby("Pump Tag").agg(
        avg_failure_probability=("failure_probability", "mean"),
        avg_days_remaining=("Time to failure (Days)", "mean")
    ).reset_index()
    # Filter to only show pump tags with avg_days_remaining <= days_threshold
    filtered = grouped[grouped["avg_days_remaining"] <= days_threshold]
    if not filtered.empty:
        print(f"Pumps likely to fail within {days_threshold} days (averages per pump):")
        print(filtered.to_string(index=False))
    else:
        print(f"No pumps predicted to fail within {days_threshold} days (after filtering by average days remaining).")
else:
    print(f"No pumps predicted to fail within {days_threshold} days.")
# ...existing


# --- Regression Model ---
def train_regression_model():
    X = df[features]
    y = df["Time to failure (Days)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    print("Regression model trained.")
    return reg

def predict_time_to_failure(reg, pump_tag):
    pump_data = df[df["Pump Tag"] == pump_tag]
    if pump_data.empty:
        return f"No data found for pump tag: {pump_tag}"
    X = pump_data[features]
    prediction = reg.predict(X)
    return prediction.mean()


# Regression
# --- Regression Model ---
def train_regression_model():
    X = df[features]
    y = df["Time to failure (Days)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    print("Regression model trained.")
    return reg

def predict_time_to_failure(reg, pump_tag):
    pump_data = df[df["Pump Tag"] == pump_tag]
    if pump_data.empty:
        return f"No data found for pump tag: {pump_tag}", None
    X = pump_data[features]
    prediction = reg.predict(X)
    # Probability estimation: here, we use the inverse of the predicted days normalized to [0,1] as a proxy
    # (lower days to failure = higher probability of failure soon)
    avg_days = prediction.mean()
    max_days = df["Time to failure (Days)"].max()
    min_days = df["Time to failure (Days)"].min()
    # Avoid division by zero
    if max_days == min_days:
        probability = 1.0 if avg_days <= max_days else 0.0
    else:
        probability = 1 - (avg_days - min_days) / (max_days - min_days)
    probability = np.clip(probability, 0, 1)
    return avg_days, probability

# Regression
reg_model = train_regression_model()
user_pump_tag = input("Enter the Pump Tag to predict time to failure: ")
predicted_days, failure_probability = predict_time_to_failure(reg_model, pump_tag=user_pump_tag)
if failure_probability is not None:
    print(f"Predicted time to failure for {user_pump_tag}: {predicted_days:.2f} days")
    print(f"Estimated probability of failure soon: {failure_probability:.2f}")
else:
    print(predicted_days)

"""


# --- OLD Code ---
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# ...existing code...

new_vibration = float(input("Enter vibration value: "))
new_X = np.array([[new_vibration]])
predicted_class = model.predict(new_X)[0]
class_probs = model.predict_proba(new_X)[0]
confidence_score = np.max(class_probs)
predicted_label = model.classes_[np.argmax(class_probs)]
#print(f"Predicted Failure: {predicted_class}")


# Interpret the predicted class
if predicted_class == 1:
    print("Failure")
else:
    print("Normal")
# ...existing code..."""
