import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# -----------------------------
# 1. Load & Preprocess Dataset
# -----------------------------
df = pd.read_csv("/content/Churn_Modelling.csv")

# Remove useless columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# One-hot encoding (drop_first avoids dummy trap)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Features & target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# -----------------------------
# 2. Train-test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn + scale
X_test_scaled = scaler.transform(X_test)         # scale only

# -----------------------------
# 4. Build ANN Model
# -----------------------------
model = Sequential([
    Dense(11, activation="relu", input_dim=11),
    Dense(6, activation="relu"),
    Dense(3, activation="relu"),
    Dense(2, activation="relu"),
    Dense(1, activation="sigmoid")      # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 5. Train the Model
# -----------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    verbose=1
)

# -----------------------------
# 6. Predictions
# -----------------------------
y_log = model.predict(X_test_scaled)
y_pred = (y_log > 0.5).astype(int)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Model Evaluate:", model.evaluate(X_test_scaled, y_test))

# -----------------------------
# 7. Plot Loss & Accuracy
# -----------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title("Training Loss")

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")

plt.show()
