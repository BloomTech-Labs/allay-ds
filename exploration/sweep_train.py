import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
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
# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparameter_defaults = dict(
    learning_rate = 0.001,
    epochs = 3,
    optimizer = "adam"
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
config = wandb.config

from process_data import reset_data_with_val
WANDB_API_KEY = "0fcba704b9f2b77c7881c3e23af2d4adf89dbbbd"

x_train, y_train, x_val, y_val = reset_data_with_val()
vect = TfidfVectorizer(stop_words = 'english', max_features=2000)
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
                callbacks=[WandbCallback(validation_data=(x_val_vect, y_val), labels=["appropriate", "inappropriate"])])


#wandb.log({'val_accuracy': accuracy})
