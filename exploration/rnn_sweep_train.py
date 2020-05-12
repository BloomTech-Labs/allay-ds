"""This module runs the training function that weights and biases fills in
with varying hyperparameters in a hyperparameter sweep.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
# pylint: disable=import-error
from tensorflow.keras.optimizers import Nadam
#from dotenv import load_dotenv
import os
from process_data import reset_data_with_val
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load data
x_train = np.load("exploration/data/x_train_sequences.npy")
x_val = np.load("exploration/data/x_val_sequences.npy")
y_train = pd.read_pickle("exploration/data/y_train.xz")
y_val = pd.read_pickle("exploration/data/y_val.xz")

# Set defaults for each parameter you are sweeping through.
hyperparameter_defaults = dict(
    learning_rate = 0.001,
    epochs = 3,
    batch_size = 128,
    weight_decay = .0005,
    dropout = .3

    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config

#load_dotenv()
#WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_API_KEY = "0fcba704b9f2b77c7881c3e23af2d4adf89dbbbd"
# Turn data into a form that Keras accepts.
# This section can be customized.
model = Sequential()
model.add(Embedding(8001,
                    128,
                    input_length=60))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(16))

model.add(Dense(units=1, activation='sigmoid'))
optimizer = Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=config.batch_size,
                epochs=config.epochs,
                validation_data=(x_val, y_val),
                callbacks=[WandbCallback(monitor='val_accuracy',
                validation_data=(x_val, y_val),
                labels=["appropriate", "inappropriate"]),
                EarlyStopping(patience=5, restore_best_weights=True)])

# Will automatically log many metrics with the callback.
#wandb.log({'val_accuracy': accuracy})
