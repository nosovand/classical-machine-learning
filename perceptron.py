import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        :param learning_rate: defines the learning rate during training
        :param n_iterations: defines the number of iterations during training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        # Declaration of weights
        self.weights = None

    def train(self, X_train, y_train):
        """
        :param X_train: The training data features
        :param y_train: The training data labels
        """
        # Initialize weights
        self.weights = np.zeros((X_train.shape[1]+1))
        # Add bias to the training features
        X_train = PolynomialFeatures(1).fit_transform(X_train)
      
        # Train the model
        for epoch in range(self.n_iterations):
            # M - number of misclassified train inputs
            M = 0
            # Iterate over the training dataset
            for X, y in zip(X_train, y_train):
                # Dot product between weights and features
                linear_output = np.dot(X, self.weights)

                # Apply step function to predict the label
                prediction = 1 if linear_output > 0 else 0

                # Update weights based on prediction error
                update = self.learning_rate * (y - prediction)
                if update != 0:
                    M+=1
                self.weights += update * X

            #End iteration when all points are correctly classified
            if M == 0:
                break

    def predict(self, X_test):
        """
        :param X_test: The test data features
        :return: Vector of predicted labels
        """
        # Predict the labels for a group of test samples
        # Add bias to the data features
        X_test = PolynomialFeatures(1).fit_transform(X_test)
        predictions = []
        # Iterate over the test samples
        for X in X_test:
            # Predict label
            linear_output = np.dot(X, self.weights)
            prediction = 1 if linear_output > 0 else 0
            # Add predicted label to the predictions
            predictions.append(prediction)
        return predictions
