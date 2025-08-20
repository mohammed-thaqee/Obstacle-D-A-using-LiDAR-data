import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess_lidar import preprocess

# --------------------------
# Load preprocessed features
# --------------------------

train_csv = "data3d/train.csv"
test_csv = "data3d/test.csv"

# Load scaler + encoder if available
try:
    scaler = joblib.load("scaler.pkl")
    relvel_encoder = joblib.load("relvel_encoder.pkl")
except:
    scaler, relvel_encoder = None, None

# Preprocess datasets
X_train, y_risk_train, y_maneuver_train, scaler, relvel_encoder = preprocess(
    train_csv, scaler=scaler, fit_scaler=True
)
X_test, y_risk_test, y_maneuver_test, _, _ = preprocess(
    test_csv, scaler=scaler
)

# Save scaler + encoder again (just to be sure)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(relvel_encoder, "relvel_encoder.pkl")

# --------------------------
# Train Models
# --------------------------

# Model 1: Collision Risk
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_risk_train)

# Model 2: Maneuver
maneuver_model = RandomForestClassifier(n_estimators=100, random_state=42)
maneuver_model.fit(X_train, y_maneuver_train)

# Save models
joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(maneuver_model, "maneuver_model.pkl")

# --------------------------
# Evaluation
# --------------------------

print("\n=== Collision Risk Evaluation ===")
y_risk_pred = risk_model.predict(X_test)
print(classification_report(y_risk_test, y_risk_pred))
print("Confusion Matrix:\n", confusion_matrix(y_risk_test, y_risk_pred))

print("\n=== Maneuver Evaluation ===")
y_maneuver_pred = maneuver_model.predict(X_test)
print(classification_report(y_maneuver_test, y_maneuver_pred))
print("Confusion Matrix:\n", confusion_matrix(y_maneuver_test, y_maneuver_pred))
