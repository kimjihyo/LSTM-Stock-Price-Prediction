import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Input, Dense, LSTM, TimeDistributed,
                                     RepeatVector, Activation)


def create_model(look_back, foward_days):
    NUM_NEURONS_FirstLayer = 128
    NUM_NEURONS_SecondLayer = 64
    # Build the model
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(
        look_back, 1), return_sequences=True))
    model.add(LSTM(NUM_NEURONS_SecondLayer,
                   input_shape=(NUM_NEURONS_FirstLayer, 1)))
    model.add(Dense(foward_days))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(model, dataX, dataY, epoch_count):
    history = model.fit(dataX, dataY, batch_size=2,
                        epochs=epoch_count, shuffle=True)


def prep_data(data, input_size, output_size):
    dX, dY = [], []
    for i in range(len(data) - input_size - output_size):
        dX.append(data[i:i + input_size])
        dY.append(data[i + input_size:i + input_size + output_size])
    return dX, dY


def split_into_train_and_test(data, train_ratio):
    train_size = int(data.shape[0] * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def generate_sin_wave():
    x = np.arange(0, 10000, step=0.1)
    y = np.sin(x)
    return y


# The number of timesteps for input
look_back = 50
# The number of timesteps to predict
look_foward = 50


# Download stock prices from yahoo finance
df = yf.Ticker('AAPL').history(interval='1m', period='1wk')
prices = df['Close'].values


# Prepare data
vals = prices
scaler = MinMaxScaler(feature_range=(-1, 1))
vals = vals.reshape(-1, 1)
vals = scaler.fit_transform(vals)
x, y = prep_data(vals, look_back, look_foward)
x = np.array(x)
y = np.array(y)


# Split data into test and train
train_x, test_x = split_into_train_and_test(x, 0.75)
train_y, test_y = split_into_train_and_test(y, 0.75)


# Create and train a model
# model = create_model(look_back, look_foward)
# train_model(model, train_x, train_y, 1)
# model.save('tsla.h5')

# Load a trained model.
model = load_model('aapl.h5')

# Make predictions using the model
# The shape of input should be a 3D array
# e.g. (1 (the number of batches), 50 (look_back), 1 (padding))
# The shape of output (prediction) will be a 2D array
# e.g. (1 (the number of batches), 10 (look_foward))
predictions = model.predict(test_x)

true = np.array([])
preds = np.array([])

# Plot a line graph with a testing dataset
for i in range(0, test_x.shape[0], look_back + look_foward):
    pre = scaler.inverse_transform(test_x[i])
    pre = pre.reshape(look_back)
    true_post = scaler.inverse_transform(test_y[i])
    true_post = true_post.reshape(look_foward)
    pred_post = scaler.inverse_transform(predictions[i].reshape(-1, 1))
    pred_post = pred_post.reshape(look_foward)
    true = np.concatenate((true, pre, true_post), axis=0)
    preds = np.concatenate(
        (preds, [None for i in range(look_back)], pred_post))
plt.plot(true)
plt.plot(preds)
plt.show()
