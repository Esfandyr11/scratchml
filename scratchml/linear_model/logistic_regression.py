import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class LogisticRegression:
    def __init__(self, lr, iter):
        self.lr = lr
        self.iter = iter
        self.loss = []
        self.history = []
        self.m = None
        self.b = None

    def forward_prop(self, X):
        pred = 1 / (1 + np.exp(-(np.dot(X, self.m) + self.b)))
        return pred

    def back_prop(self, X, Y, pred):
        diff = (pred - Y)
        dw = np.mean(np.multiply(X, diff.reshape(-1, 1)), axis=0)
        db = np.mean(diff)
        return dw, db

    def train(self, X, Y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.iter):
            pred = self.forward_prop(X)
            dw, db = self.back_prop(X, Y, pred)
            
            self.m -= self.lr * dw
            self.b -= self.lr * db
            
            loss = -np.mean(Y * np.log(pred + 1e-15) + (1 - Y) * np.log(1 - pred + 1e-15))
            print(f"Iterations : {_}, Loss Value : {loss}")
            self.loss.append(loss)
            self.history.append((self.m.copy(), self.b))

    def animate(self, X, Y):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', edgecolors='black')
        contour = ax.contourf(xx, yy, np.zeros_like(xx), levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.3)
        line, = ax.plot([], [], 'k-', lw=2)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Logistic Regression Decision Boundary')
        
        def update(frame):
            m, b = self.history[frame]
            Z = 1 / (1 + np.exp(-(xx * m[0] + yy * m[1] + b)))
            
            for c in ax.collections[1:]:
                c.remove()
            
            contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.3)
            x_line = np.array([x_min, x_max])
            y_line = -(m[0] * x_line + b) / m[1]
            line.set_data(x_line, y_line)
            
            return scatter, contour, line
        
        anim = FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=False)
        
        writer = PillowWriter(fps=25)
        anim.save('logistic_regression.gif', writer=writer)
        plt.show()
        plt.close(fig)
        print("Animation saved as 'logistic_regression.gif'")

