import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class QuadraticRegression:
    def __init__(self, iter=1000, lr=0.01):
        self.m1 = None
        self.m2 = None
        self.b = None
        self.loss = []
        self.history = []
        self.iter = iter
        self.lr = lr
    
    def prediction(self, X):
        return self.m2 * X**2 + self.m1 * X + self.b
    
    def cost_func(self, Y, pred):
        diff = pred - Y
        cost = (0.5) * np.mean(diff**2)
        return cost
    
    def back_prop(self, X, Y, pred):
        diff = pred - Y
        db = np.mean(diff)
        dw1 = np.mean(X * diff)
        dw2 = np.mean(X**2 * diff)
        return dw2, dw1, db
    
    def train(self, X, Y):
        self.b = 0
        self.m1 = 0
        self.m2 = 0

        # Normalize data
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_normalized = (X - X_mean) / X_std

        for i in range(self.iter):
            y_pred = self.prediction(X_normalized)
            cost = self.cost_func(Y, y_pred)
            self.loss.append(cost)
            
            dw2, dw1, db = self.back_prop(X_normalized, Y, y_pred)
            self.m1 -= self.lr * dw1
            self.m2 -= self.lr * dw2
            self.b -= self.lr * db
            print(f"iterations: {i+1}, loss value: {self.loss[i]}")
            
            self.history.append((self.m2, self.m1, self.b))
            
        return self
    
    def animate(self, X, Y, save_path='quadratic_regression.gif'):
            # Create a figure and axis for the animation
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, Y, color='blue')
            line, = ax.plot(X, self.prediction((X - np.mean(X)) / np.std(X)), color='red')

            # Function to update the animation
            def update(i):
                m2, m1, b = self.history[i]
                line.set_ydata(m2 * ((X - np.mean(X)) / np.std(X))**2 + m1 * ((X - np.mean(X)) / np.std(X)) + b)
                return line,

            # Create the animation
            ani = FuncAnimation(fig, update, frames=len(self.history), blit=True)

            # Save the animation as a GIF
            ani.save(save_path, writer=PillowWriter(fps=20))
            plt.title('Quadratic Regression')
            # Display the animation
            plt.show()
