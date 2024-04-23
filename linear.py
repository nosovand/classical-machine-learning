import numpy as np

class LinearRegression:
    def __init__(self):
        '''
        Constructor for the LinearRegression class
        ''' 
        self.X_train = []
        self.y_train = []
        self.weights = []
        
    def fit(self, X, y):
        '''
        Fit the model to the data.
        :param X: The training data
        :param y: The training labels
        '''
        self.X_train = X
        self.y_train = y
        # Using closed-form solution to calculate the weights
        self.weights = np.linalg.solve(X.T@X,X.T@y)
    
    def predict(self,x_test):
        '''
        Predict using the linear regression model.
        :param X: The data to predict on
        :return: The predicted labels
        '''
        self.y_hat=np.sum(x_test*self.weights,axis=1)
        
        return self.y_hat
    
    def MSE(self,y_pred, y_test) :
        '''
        Calculate the mean squared error.
        :param y_pred: The predicted labels
        :param y_test: The true labels
        :return: The mean squared error
        '''
        MSE = 1.0/len(y_pred) * np.sum(np.square(np.subtract(y_test, y_pred)))
        return MSE