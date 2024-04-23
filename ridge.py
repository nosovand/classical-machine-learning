'''
This file contains a RidgeRegression class, which is used to perform ridge regression.
'''

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class RidgeRegression:
    '''
    This class performs ridge regression using the closed-form solution.
    '''
    def __init__(self, p_lambda=1.0):
        '''
        Constructor for the RidgeRegression class.
        :param p_lambda: The regularization parameter lambda
        '''
        self.p_lambda = p_lambda

    def fit(self, X, y):
        '''
        Fit the model to the data.
        :param X: The training data
        :param y: The training labels
        '''
        # Get number of samples and features
        n_samples, n_features = X.shape
        # Normalize X
        self.mean_X = X.mean(axis=0)
        self.std_X = X.std(axis=0)
        X_normalized = (X - self.mean_X) / self.std_X
        # Add bias term
        X_normalized = np.hstack([np.ones((n_samples, 1)), X_normalized])
        # Normalize y
        self.mean_y = y.mean()
        y_normalized = y - self.mean_y
        # Create identity matrix for regularization, excluding bias term
        identity_matrix = np.identity(n_features + 1)
        # Calculate coefficients using closed-form solution
        self.coefficients = np.linalg.inv(X_normalized.T @ X_normalized + self.p_lambda * identity_matrix) @ X_normalized.T @ y_normalized

    def predict(self, X):
        '''
        Predict using the ridge regression model.
        :param X: The data to predict on
        :return: The predicted labels
        '''
        # Apply the same normalization as used in fit to X
        X_normalized = (X - self.mean_X) / self.std_X
        # Add bias term
        X_normalized = np.hstack([np.ones((X.shape[0], 1)), X_normalized])
        # Predict
        # Add back the mean of y to the prediction
        return X_normalized @ self.coefficients + self.mean_y

    def get_coefficients(self):
        '''
        Get the coefficients of the model.
        :return: The coefficients
        '''
        return self.coefficients
