import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 1) * 100  # Feature: Random numbers between 0-100
y = 3.5 * X + np.random.randn(1000, 1) * 10  # Linear equation with noise

# Convert to DataFrame
df = pd.DataFrame(np.hstack((X, y)), columns=["Feature", "Target"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[["Feature"]], df["Target"], test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the DNN Model
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),  # Hidden Layer
    Dense(5, activation='relu'),  # Another Hidden Layer
    Dense(1)  # Output Layer (Linear activation by default)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Linear Regression using Deep Neural Network")
plt.show()
