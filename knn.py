import numpy as np

class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''     
        self.X = X
        self.y = y   
       
    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        distances = self.minkowski_dist(X_new, p)
        sorted_indices = np.argsort(distances, axis=1)
        nearest_indices = sorted_indices[:, :self.k]
        nearest_labels = self.y[nearest_indices]
        y_hat = []
        for labels in nearest_labels:
            (values, counts) = np.unique(labels, return_counts=True)
            index = np.argmax(counts)
            y_hat.append(values[index])
        return np.array(y_hat)

            

    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance

        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        if p == 0:
            raise ValueError("p cannot be zero")
        dst = np.empty((X_new.shape[0], self.X.shape[0]))
        for i, x in enumerate(X_new):
            dst[i, :] = np.sum(np.abs(self.X - x) ** p, axis=1) ** (1 / p)
        return dst




