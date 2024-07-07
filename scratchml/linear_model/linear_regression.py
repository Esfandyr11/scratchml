import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class LinearRegression:
    """initializing parameters learning rate and the number of iterations , also m variable which 
    
    refrernces the weights i.e the slopes of the multi regression , while b is the bias in the regression

    , to display the loss over iterations, also the weights and biases are recorded in a array named history"""
    def __init__(self, lr=0.1, iter=1000):
        self.lr = lr
        self.iter = iter
        self.m = None
        self.b = None
        self.loss = []
        self.history = []
        
    """the prediction function works both for well, predictions but in this module it will also function 

        as the forward propagation function"""
    
    def prediction(self, X):
        pred = np.dot(X, self.m) + self.b
        return pred

    """the mean squared error loss function"""
    def cost_func(self, Y, pred):
        cost = np.mean((Y - pred) ** 2)
        return cost

    """ a back propagation gradient descent function J """
    def back_prop(self, X, Y, pred):
        diff = (pred - Y)
        dw = np.mean(np.multiply(X, diff.reshape(-1, 1)), axis=0)
        db = np.mean(diff)
        return dw, db
    
    def train(self, X, Y):
        n_tokens, n_feature = X.shape
        self.b = 0
        self.m = np.random.rand(n_feature)

        for _ in range(self.iter):
            y_pred = self.prediction(X)
            cost = self.cost_func(Y, y_pred)
            self.loss.append(cost)
            
            dw, db = self.back_prop(X, Y, y_pred)
            self.m = self.m - self.lr * dw
            self.b = self.b - self.lr * db
            print(f"iterations : {_+1} , loss value : {self.loss[_]}")
            # Store the history of m and b for plotting
            self.history.append((self.m.copy(), self.b))
            
        return self
    
    def animate(self, X, Y, save_path='linear_regression.gif'):
        # Create a figure and axis for the animation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, Y, color='blue')
        line, = ax.plot(X, self.prediction(X), color='red')

        # Function to update the animation
        def update(i):
            m, b = self.history[i]
            line.set_ydata(np.dot(X, m) + b)
            return line,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(self.history), blit=True)

        # Save the animation as a GIF
        ani.save(save_path, writer=PillowWriter(fps=10))
        plt.title('Linear Regression')
        # Display the animation
        plt.show()
