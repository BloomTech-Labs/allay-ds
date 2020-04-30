"""This module runs the training function that weights and biases fills in
with varying hyperparameters in a hyperparameter sweep.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
# pylint: disable=import-error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Nadam
#from dotenv import load_dotenv
import os
from process_data import reset_data_with_val

# Set defaults for each parameter you are sweeping through.
hyperparameter_defaults = dict(
    learning_rate = 0.001,
    epochs = 3,
    optimizer = "adam"
    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config

# This needs to be an environment variable
WANDB_API_KEY = INSERT_API_KEY_HERE

# Turn data into a form that Keras accepts.
# This section can be customized.
x_train, y_train, x_val, y_val = reset_data_with_val()
vect = TfidfVectorizer(stop_words='english', max_features=2000)
x_train_vect = vect.fit_transform(x_train)
x_val_vect = vect.transform(x_val)
x_train_vect = x_train_vect.toarray()
x_val_vect = x_val_vect.toarray()

model = Sequential()

model.add(Dense(128, input_dim=2000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = config.optimizer, metrics=['accuracy'])

model.fit(x_train_vect, y_train, batch_size=32,
                steps_per_epoch=len(x_train_vect) / 32, epochs=config.epochs,
                validation_data=(x_val_vect, y_val),
                callbacks=[WandbCallback(validation_data=(x_val_vect, y_val),
                labels=["appropriate", "inappropriate"])])

# Will automatically log many metrics with the callback.
#wandb.log({'val_accuracy': accuracy})
