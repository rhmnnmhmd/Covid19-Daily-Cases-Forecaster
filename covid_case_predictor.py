# # Libraries
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

from modules import create_model, plot_loss_metric, plot_predictions, print_performance_metrics


# # Statics
# model path
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')

# data path
DATA_PATH_TRAIN = os.path.join(os.getcwd(), 'data', 'cases_malaysia_train.csv')
DATA_PATH_TEST = os.path.join(os.getcwd(), 'data', 'cases_malaysia_test.csv')

# scaler path
SCALER_PATH = os.path.join(os.getcwd(), 'model', 'scaler.pkl')

# logs path
time_stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(), 'logs', time_stamp)

# # Data loading
df_train = pd.read_csv(DATA_PATH_TRAIN)
df_test = pd.read_csv(DATA_PATH_TEST)

# ## general infos
df_train.info()
df_test.info()

# # Data wrangling
# ## change date to datetime format
df_train['date'] = pd.to_datetime(df_train['date'], format='%d/%m/%Y')
df_test['date'] = pd.to_datetime(df_test['date'], format='%d/%m/%Y')

# ## change from 'object' data type to 'numeric' data type (for 'cases_new' column in df_train)
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')

# ## interpolation 
# ### for train data
df_train['cases_new'] = df_train['cases_new'].interpolate(method='pad')

# ### for test data
df_test['cases_new'] = df_test['cases_new'].interpolate(method='pad')

# # Basic plotting of the data
# ## train data
fig, ax =plt.subplots(1, 1, figsize=(10, 5))
df_train['cases_new'].plot.line(ax=ax)
ax.set_title('Covid-19 Daily New Cases (Jan 2020 - Dec 2021)')
ax.set_ylabel('new cases')
ax.set_xlabel('days since 25 Jan 2020')
plt.tight_layout()
plt.show()

# ## test data
fig, ax =plt.subplots(1, 1, figsize=(10, 5))
df_test['cases_new'].plot.line(ax=ax)
ax.set_title('Covid-19 Daily New Cases (Dec 2021 - Mar 2022)')
ax.set_ylabel('new cases')
ax.set_xlabel('days since 5 Dec 2021')
plt.tight_layout()
plt.show()

# # Min-max scale the number of new cases
# ## for train data
mms = MinMaxScaler()
cases_new_train = mms.fit_transform(df_train[['cases_new']])

# ## for test data
cases_new_test = mms.transform(df_test[['cases_new']])

# ## save the min-max scaler
with open(SCALER_PATH, 'wb') as file:
    pickle.dump(mms, file)

# # Combine cases_new_train and cases_new_test for later usage
to_be_concat = (cases_new_train, cases_new_test)
cases_new_combined = np.concatenate(to_be_concat, axis=0)[-130: ]

# # Create features and target
# ## for train data
window_size = 30
X_train = []
y_train = []

for i in range(window_size, cases_new_train.shape[0]):
    X_train.append(cases_new_train[i-window_size: i, 0])
    y_train.append(cases_new_train[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# ## for test data
X_test = []

for i in range(window_size, cases_new_combined.shape[0]):
    X_test.append(cases_new_combined[i-window_size: i, 0])

X_test = np.array(X_test)
y_test = np.copy(cases_new_test)

# # Deep learning
# ## create model
input_shape = X_train.shape[-1]
output_shape = 1
n_lstm = 64
drop_rate_1 = 0.3
drop_rate_2 = 0.2

model = create_model(input_shape=input_shape, 
                     output_shape=output_shape, 
                     n_lstm=n_lstm, 
                     drop_rate_1=drop_rate_1, 
                     drop_rate_2=drop_rate_2)

# ## model summary
model.summary()

# ## model plot
plot_model(model, show_shapes=True, show_layer_names=True)

# ## compile model
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=[mean_absolute_percentage_error])

# ## callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1)

# ## model fitting/training with History object
model_hist = model.fit(X_train, 
                       y_train, 
                       epochs=100,
                       batch_size=65,
                       callbacks=[tensorboard_callback, early_stopping_callback])

# ## plot the loss and metrics
plot_loss_metric(model_hist)

# # Predictions
y_pred = model.predict(np.expand_dims(X_test, axis=-1))
y_pred_actual = mms.inverse_transform(y_pred)
y_test_actual = mms.inverse_transform(y_test)

plot_predictions(y_test_actual, y_pred_actual)

# # Performance metrics
print_performance_metrics(y_test_actual, y_pred_actual)

# # Save the best model
model.save(MODEL_PATH)

# # Tensorboard run in terminal
# %load_ext tensorboard
# %tensorboard --logdir logs