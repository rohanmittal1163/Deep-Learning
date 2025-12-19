# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 2. Load and Explore Data
# -----------------------------
df = pd.read_csv("/content/Admission_Predict.csv")

# Drop unnecessary column
df = df.drop(columns=['Serial No.'])

# Features and target
X = df.drop(columns=['Chance of Admit '])
y = df['Chance of Admit ']

# Check for duplicates
print("Duplicates:", df.duplicated().sum())

# -----------------------------
# 3. Visualize Skew and Distribution
# -----------------------------
skew_list = []
for i, col in enumerate(X.columns):
    plt.figure()
    plt.hist(X[col], bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
    skew_list.append(X[col].skew())

print("Skew values:", skew_list)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Feature Scaling
# -----------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn + transform
X_test_scaled = scaler.transform(X_test)         # Only transform

# -----------------------------
# 6. Build Regression Model
# -----------------------------
model = Sequential([
    Dense(7, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(1, activation='linear')  # Regression output
])

# Compile model with regression loss
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']  # better for regression
)

model.summary()

# -----------------------------
# 7. Train the Model
# -----------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled).reshape(-1)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------
# 9. Plot Training History
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title("Mean Absolute Error")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.show()
