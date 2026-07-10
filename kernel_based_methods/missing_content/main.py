import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.RandomState(42)
X_train = 0.3 * rng.randn(100, 2)
X_train = np.r_[X_train + 2, X_train - 2]
X_test = np.r_[0.3 * rng.randn(20, 2) + 2,
               rng.uniform(low=-6, high=6, size=(10, 2))]

# 1) Initialize One-Class SVM
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)

# 2) Fit on training data
model.fit(X_train)

# 3) Predict on test data
preds = model.predict(X_test)

# 4) Compute fraction of anomalies
anomaly_fraction = np.mean(preds == -1)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Detected anomalies:", np.sum(preds == -1))
print("Anomaly fraction:", anomaly_fraction)