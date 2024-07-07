import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class LassoRegression:

    def __init__(self,iter,lr,l1):
        self.lr = lr
        self.iter = iter
        self.l1 = l1
        self.m = None
        self.b = None
        self.loss = []
        self.history =[]
    
    def prediction(self,X):
        pred = np.dot(X,self.m)+self.b
        return pred
    
    def cost_func(self,Y, pred):
        diff = Y-pred
        cost = np.mean(diff**2)+self.l1*np.sum(np.abs(self.m))
        return cost
    
    def back_prop(self,Y,pred,X):
        diff = Y-pred
        dw = -2 * np.dot(X.T,diff)/X.shape[0] + self.l1*np.sign(self.m)
        db = -2 * np.mean(diff)
        return dw, db

    def train(self,X,Y):
        n_samples, n_features = X.shape
        self.b = 0
        self.m = np.random.rand(n_features, 1)  
        
        for i in range(self.iter):
            y_pred = self.prediction(X)
            cost = self.cost_func(Y, y_pred)
            dw, db = self.back_prop(Y, y_pred, X)
            self.m -= self.lr * dw
            self.b -= self.lr * db
            
            self.loss.append(cost)
            print(f"Iteration: {i + 1}, Loss: {cost}")
            self.history.append((self.m.copy(),self.b))
        return self
    def animate(self, X, Y, save_path='lasso_regression.gif'):
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
        plt.title('Lasso Regression')
        # Display the animation
        plt.show()

