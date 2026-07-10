import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Given synthetic data (do not modify)
rng = np.random.RandomState(42)
X_inliers = 0.3 * rng.randn(100, 2)
X_inliers = np.r_[X_inliers + 3, X_inliers - 3]
X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack((X_inliers, X_outliers))

# 1) Initialize LOF model
model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# 2) Fit and predict
preds = model.fit_predict(X)

# 3) Extract negative outlier factor
scores = model.negative_outlier_factor_

# 4) Print results
print("Total samples:", len(X))
print("Detected outliers:", np.sum(preds == -1))
print("Example LOF scores:", scores[:5])