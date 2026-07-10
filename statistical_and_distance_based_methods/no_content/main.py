import numpy as np

X = np.array([
    [4.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [8.0, 7.0],
    [9.0, 6.0],
    [3.5, 2.5],
])

# 1) Mean vector
mean_vec = X.mean(axis=0)

# 2) Covariance matrix and inverse
cov = np.cov(X.T)
cov_inv = np.linalg.inv(cov)

# 3) Mahalanobis distances
diff = X - mean_vec
distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

# 4) Outlier mask
threshold = 2.5
outliers = distances > threshold

print("Mean vector:", mean_vec)
print("Covariance matrix:\n", cov)
print("Mahalanobis distances:", distances)
print("Outliers mask:", outliers)