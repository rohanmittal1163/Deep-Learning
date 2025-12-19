import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

data = pd.read_csv("/content/placement.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

p = Perceptron()
p.fit(X,y)
print(p.predict([[90,80]]))
print(p.intercept_)
print(p.coef_)
print(p.score(X,y))

plot_decision_regions(X,y,clf=p,legend=2)
plt.show()