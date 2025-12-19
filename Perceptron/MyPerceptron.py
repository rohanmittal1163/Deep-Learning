import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

class MyPerceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Convert labels to {-1, +1} for perceptron loss
        y_ = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Shuffle data each epoch (sklearn does this)
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                xi = X[i]
                yi = y_[i]

                condition = yi * (np.dot(xi, self.weights) + self.bias)

                # Perceptron loss: update only if misclassified
                if condition <= 0:
                    self.weights += self.lr * yi * xi
                    self.bias   += self.lr * yi

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear >= 0, 1, 0)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# Load data
data = pd.read_csv("/content/placement.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train
p = MyPerceptron(lr=0.01, epochs=1000)
p.fit(X, y)

# Test
print("Prediction:", p.predict([[90, 80]]))
print("Accuracy:", p.score(X, y))

# Plot
plot_decision_regions(X, y, clf=p)
plt.show()
