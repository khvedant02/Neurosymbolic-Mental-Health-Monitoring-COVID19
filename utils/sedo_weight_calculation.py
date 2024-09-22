# utils/sedo_weight_calculation.py

import numpy as np
from scipy import linalg
import pickle

def SAE(X, S, lamb):
    A = S.dot(S.T)  # Self-correlation between categories
    B = lamb * (X.dot(X.T))  # Self-correlation tweet
    C = (1 + lamb) * (S.dot(X.T))  # Cross-correlation
    W = linalg.solve_sylvester(A, B, C)
    return W

def train_sedo(X_train, Y_train, S_train, lamb=0.12):
    W = np.linalg.inv(np.eye(X_train.T.dot(X_train).shape[0]) * 50 + X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
    X_train = X_train.dot(W)
    S_train = S_train.dot(W)
    W = SAE(X_train, S_train.T, lamb).T
    return W

# Example usage within this module (for module testing purposes only)
if __name__ == "__main__":
    X_train = np.random.rand(10, 300)
    Y_train = np.random.rand(10, 3)
    S_train = np.random.rand(3, 300)
    W = train_sedo(X_train, Y_train, S_train)
    print(W)
