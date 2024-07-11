import numpy as np

class DataSplitter:
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
    
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X, Y):
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            
            X = X[indices]
            Y = Y[indices]
        
        test_size = int(self.test_size * X.shape[0])
        x_train, x_test = X[test_size:], X[:test_size]
        y_train, y_test = Y[test_size:], Y[:test_size]
        
        return x_train, x_test, y_train, y_test