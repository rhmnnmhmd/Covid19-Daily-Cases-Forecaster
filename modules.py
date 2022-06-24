import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error 

import matplotlib.pyplot as plt

import os

import datetime as dt

import seaborn as sns

import pandas as pd

import sklearn 
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pickle


def create_model(input_shape, output_shape, n_lstm, drop_rate_1, drop_rate_2):
    input_1 = Input(shape=(input_shape, 1))

    hl_1 = LSTM(n_lstm, return_sequences=True)(input_1)
    hl_2 = Dropout(drop_rate_1)(hl_1)
    hl_3 = LSTM(n_lstm)(hl_2)
    hl_4 = Dropout(drop_rate_2)(hl_3)

    output_1 = Dense(output_shape, activation='selu', kernel_initializer='lecun_normal')(hl_4)

    return Model(inputs=input_1, outputs=output_1)


def plot_loss_metric(model_hist):
    train_loss = model_hist.history['loss']
    train_metrics = model_hist.history['mean_absolute_percentage_error']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('MSE Loss and MAPE Metrics vs Epochs')

    ax[0].plot(train_loss)
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('MSE loss')

    ax[1].plot(train_metrics)
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('MAPE metrics')

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(y_true, label='Actual new cases')
    ax.plot(y_pred, label='Predicted new cases')
    ax.set_title('Covid-19 Daily New Cases (Dec 2021 - Mar 2022)')
    ax.set_ylabel('new cases')
    ax.set_xlabel('days since 5 Dec 2021')
    ax.legend()

    plt.tight_layout()
    plt.show()


def print_performance_metrics(y_test_actual, y_pred_actual):
    mse = mean_squared_error(y_test_actual[0: , 0], y_pred_actual[0: , 0])
    mae = mean_absolute_error(y_test_actual[0: , 0], y_pred_actual[0: , 0])
    mape = mean_absolute_percentage_error(y_test_actual[0: , 0], y_pred_actual[0: , 0])

    print(f'MSE is {mse}\nMAE is {mae}\nMAPE is {mape}')