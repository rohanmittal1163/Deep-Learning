import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load & Normalize Data
# ----------------------------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# ----------------------------------------------------
# 2. Build ANN Model
# ----------------------------------------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Converts 28x28 to 784
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")      # 10 classes (0–9 digits)
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------------------------
# 3. Train the Model
# ----------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------------------------
# 4. Predictions
# ----------------------------------------------------
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

# ----------------------------------------------------
# 5. Plot Training Curves
# ----------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(history.history['loss'])
plt.title("Training Loss")

plt.subplot(1, 4, 2)
plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")

plt.subplot(1, 4, 3)
plt.plot(history.history['val_loss'])
plt.title("Validation Loss")

plt.subplot(1, 4, 4)
plt.plot(history.history['val_accuracy'])
plt.title("Validation Accuracy")

plt.tight_layout()
plt.show()
