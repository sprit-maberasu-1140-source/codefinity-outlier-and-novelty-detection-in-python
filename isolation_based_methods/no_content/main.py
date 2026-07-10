import numpy as np
from sklearn.ensemble import IsolationForest

# Given synthetic data (do not modify)
rng = np.random.RandomState(42)
X_inliers = 0.3 * rng.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]  # two clusters
X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack((X_inliers, X_outliers))

# 1) Initialize Isolation Forest
model = IsolationForest(contamination=0.15, n_estimators=100, random_state=42)

# 2) Fit model on the data
model.fit(X)

# 3) Compute anomaly scores
scores = model.decision_function(X)

# 4) Predict inliers/outliers
preds = model.predict(X)

# 5) Print key outputs
print("Total samples:", len(X))
print("Detected outliers:", np.sum(preds == -1))
print("Example anomaly scores:", scores[:5])