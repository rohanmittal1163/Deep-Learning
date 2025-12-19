import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class Perceptron:
    def __init__(self, lr=0.001, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def step(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        # Add bias term
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.ones(n_features)

        # Perceptron learning rule
        for _ in range(self.epochs):
            idx = np.random.randint(0, n_samples)
            x_i = X[idx]
            y_i = y[idx]

            prediction = self.step(np.dot(x_i, self.weights))
            error = y_i - prediction

            # Weight update
            self.weights += self.lr * error * x_i

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        linear_output = np.dot(X, self.weights)
        return np.where(linear_output >= 0, 1, 0)

    def coef_(self):
        return self.weights[1:]

    def intercept_(self):
        return self.weights[0]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def plot(self, X, y):
        plot_decision_regions(X, y, clf=self, legend=2)
        plt.title("Perceptron Decision Boundary")
        plt.show()


# ------------------------------
# Load Data & Train Perceptron
# ------------------------------

data = pd.read_csv("/content/placement.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

p = Perceptron(lr=0.001, epochs=10000)
p.fit(X, y)

print("Prediction for [90, 80]:", p.predict([[90, 80]]))
print("Intercept:", p.intercept_())
print("Coefficients:", p.coef_())
print("Accuracy:", p.score(X, y))

p.plot(X, y)
