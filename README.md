# Project1
i developed this stock prediction google using python yfinanceframework
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Fetch Google stock price data
google = yf.download('GOOGL', start='2010-01-01', end='2023-01-01')

# Select the 'Close' prices
data = google[['Close']]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences and labels
sequence_length = 30
sequences = []
labels = []
for i in range(len(data_scaled) - sequence_length):
    sequences.append(data_scaled[i:i+sequence_length])
    labels.append(data_scaled[i+sequence_length])

sequences = np.array(sequences)
labels = np.array(labels)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(sequences))

train_sequences = sequences[:split_index]
train_labels = labels[:split_index]
test_sequences = sequences[split_index:]
test_labels = labels[split_index:]

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences, train_labels, epochs=50, batch_size=64, verbose=1)

# Predict stock prices
predictions = model.predict(test_sequences)
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(test_labels)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(google.index[-len(actual_prices):], actual_prices, label='Actual Prices')
plt.plot(google.index[-len(predictions):], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show()
