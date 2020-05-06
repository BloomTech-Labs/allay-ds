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
    epochs = 3
    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config

#load_dotenv()
#WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_API_KEY = "0fcba704b9f2b77c7881c3e23af2d4adf89dbbbd"
# Turn data into a form that Keras accepts.
# This section can be customized.
model = Sequential()
model.add(Embedding(5001,
                    64,
                    input_length=60))
model.add(LSTM(64))

#model.add(Dropout(.2))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(8, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32,
                steps_per_epoch=len(x_train) / 32, epochs=config.epochs,
                validation_data=(x_val, y_val),
                callbacks=[WandbCallback(validation_data=(x_val, y_val),
                labels=["appropriate", "inappropriate"])])

# Will automatically log many metrics with the callback.
#wandb.log({'val_accuracy': accuracy})
