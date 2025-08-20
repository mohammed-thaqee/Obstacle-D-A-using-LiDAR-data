import os, json, math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# -------------------------
# Feature Extraction Utils
# -------------------------

def compute_features(point_cloud):
    """
    Given a LiDAR scan as list of (x, y, z) points,
    compute min_dist, min_angle, and sector-wise min distances.
    """
    if len(point_cloud) == 0:
        # No obstacle detected
        return {
            "min_dist": 30.0,
            "min_angle": 0.0,
            "sector_min_dists": [30.0] * 5
        }

    # Convert to numpy for easier ops
    pts = np.array(point_cloud)
    dists = np.linalg.norm(pts, axis=1)

    # Find nearest obstacle
    min_idx = np.argmin(dists)
    min_dist = float(dists[min_idx])
    min_angle = float(np.degrees(np.arctan2(pts[min_idx,1], pts[min_idx,0])))

    # Sectorization: split azimuth [-90°, +90°] into 5 bins
    azimuths = np.degrees(np.arctan2(pts[:,1], pts[:,0]))
    bins = np.linspace(-90, 90, 6)  # 5 sectors
    sector_min_dists = []
    for i in range(5):
        mask = (azimuths >= bins[i]) & (azimuths < bins[i+1])
        if np.any(mask):
            sector_min_dists.append(float(np.min(dists[mask])))
        else:
            sector_min_dists.append(30.0)  # no points → max range

    return {
        "min_dist": min_dist,
        "min_angle": min_angle,
        "sector_min_dists": sector_min_dists
    }

# -------------------------
# Main Preprocessing
# -------------------------

def preprocess(csv_path, scaler=None, fit_scaler=False):
    df = pd.read_csv(csv_path)

    X_features = []
    for _, row in df.iterrows():
        pts = json.loads(row["lidar_scan_data"])
        feats = compute_features(pts)
        feat_vector = [feats["min_dist"], feats["min_angle"]] + feats["sector_min_dists"]
        feat_vector.append(row["relative_velocity"])  # categorical for now
        X_features.append(feat_vector)

    X = pd.DataFrame(X_features,
                     columns=["min_dist", "min_angle",
                              "sector_leftfar", "sector_left",
                              "sector_front", "sector_right",
                              "sector_rightfar", "relative_velocity"])

    # Encode relative_velocity
    le_vel = LabelEncoder()
    X["relative_velocity"] = le_vel.fit_transform(X["relative_velocity"])

    # Targets
    y_risk = df["collision_risk"].values
    y_maneuver = df["recommended_maneuver"].values

    # Scale numeric features
    num_cols = ["min_dist", "min_angle", "sector_leftfar", "sector_left",
                "sector_front", "sector_right", "sector_rightfar"]
    if scaler is None:
        scaler = StandardScaler()
        if fit_scaler:
            X[num_cols] = scaler.fit_transform(X[num_cols])
        else:
            X[num_cols] = scaler.transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])

    return X.values, y_risk, y_maneuver, scaler, le_vel

if __name__ == "__main__":
    # Paths
    train_csv = "data3d/train.csv"
    test_csv = "data3d/test.csv"

    # Preprocess train (fit scaler)
    X_train, y_risk_train, y_maneuver_train, scaler, le_vel = preprocess(train_csv, fit_scaler=True)

    # Save scaler + encoder
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_vel, "relvel_encoder.pkl")

    # Preprocess test (use same scaler)
    X_test, y_risk_test, y_maneuver_test, _, _ = preprocess(test_csv, scaler=scaler)

    print("Train features:", X_train.shape)
    print("Test features:", X_test.shape)
