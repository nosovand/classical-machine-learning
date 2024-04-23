import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LogisticRegression:
    
    def initialize_weights(self,X):
        '''
        Initializes the parameters so that the have the same dimensions as the input data + 1
        to account for w_0 (bias)
        Inputs:
        X - input data matrix of dimensions N x D
        
        Outputs:
        weights - model parameters initialized to zero size (D + 1) x 1
        '''
        weights = np.zeros((X.shape[1]+1,1))
        
        return weights
    
    def initialize_X(self,X):
        '''
        Reshapes the input data so that it can handle w_0 (bias)
        Inputs:
        X - input data matrix of dimensions N x D
        Outputs:
        X - matrix of size N x (D + 1)
        '''
        X = PolynomialFeatures(1).fit_transform(X) #Adds a one to the matrix so it copes with w_0
        
        return X
    
    def sigmoid(self,z):
        '''
        Implements the sigmoid function
        Input:
        z - input variable 
        
        Output:
        1/(1+exp(-z))
        '''
        sig = np.divide(1,1 + np.exp(-z))
        return sig
        
    def loss_function(self,X,y,w):
        '''
        Implements the cross-entropy loss function for logistic regression

        Input:
        X - Input matrix of size N x (D + 1)
        y - Label vector of size N x 1
        w - Parameters vector of size (D + 1) x 1
        
        Output: 
        Estimation of the cross-entropy loss given the input, labels and parameters (scalar value)
        '''
        
        # 1) Estimate Xw
        Xw = np.dot(X, w)

        # 2) Estimate sigmoid of Xw
        s_xw = self.sigmoid(Xw)

        # 3) Estimate log(sig) and log(1-sig)
        logs = np.log(s_xw)
        log1s = np.log(1 - s_xw)

        # 4) Combine point 3 with the labels and sum over all elements to obtain the final estimate
        loss = -np.sum(y * logs + (1 - y) * log1s)
        
        return loss
    
    def gradient_descent_step(self,X, y, w, alpha):
        '''
        Implements a gradient descent step for logistic regression
        Input:
        X - Input matrix of size N x (D + 1)
        y - Label vector of size N x 1
        w - Parameters vector of size (D + 1) x 1
        alpha - Learning rate 
        Output: 
        Updated weights
        '''
        
        w = w + (alpha/y.shape[0])*np.dot(np.transpose(X), (y-self.sigmoid(np.dot(X, w))))
        
        return w
    
    def fit(self,X,y,alpha=0.01,iter=10, epsilon = 0.0001):
        '''
        Learning procedure of the logistic regression model
        Input:
        X - Input matrix of size N x (D + 1)
        y - Label vector of size N x 1
        alpha - Learning rate (default value 0.01)
        iter - Number of iterations to perform for gradient descent (default 10)
        epsilon - stopping criterion (default 0.0001)
        Output: 
        List of values of the loss function during the gradient descent iterations
        '''
        weights = self.initialize_weights(X) #Initializes the weights of the model
        X = self.initialize_X(X) #reformats X
        
        
        loss_list = np.zeros(iter,) # Stores the loss function values
        
        for i in range(iter):
            weights = self.gradient_descent_step(X, y, weights, alpha)
            
            loss_list[i] = self.loss_function(X,y,weights)
            
            if loss_list[i] <= epsilon:
                break
            
        self.weights = weights
        
        return loss_list
    
    def predict(self,X):
        '''
        Predicts labels y given an input matrix X
        Input: 
        X- matrix of dimensions N x D
        
        Output:
        y_pred - vector of labels (dimensions N x 1)
        '''
        #1) Reformat the matrix X
        X = self.initialize_X(X)
        
        #2) Estimate Xweights
        a = np.dot(X, self.weights)

        #3) Estimate the predicted labels
        y_pred = a > 0
        
        return y_pred.astype(int)
        