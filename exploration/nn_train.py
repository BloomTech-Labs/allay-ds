'''
This module is created to hold the neural network training function
that is necessary to run weights and biases hyperparameter sweeps.
https://github.com/wandb/client/issues/956
per the above issue.
'''
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

#load_dotenv()
#WANDB_API_KEY = os.getenv("WANDB_API_KEY")

from process_data import reset_data_with_val
WANDB_API_KEY = "0fcba704b9f2b77c7881c3e23af2d4adf89dbbbd"


x_train, y_train, x_val, y_val = reset_data_with_val()
vect = TfidfVectorizer(stop_words = 'english', max_features=2000)
x_train_vect = vect.fit_transform(x_train)
x_val_vect = vect.transform(x_val)
x_train_vect = x_train_vect.toarray()
x_val_vect = x_val_vect.toarray()

sweep_config = {
    'method': 'grid', #grid, random, bayes
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [3, 6]
        },
        'weight_decay': {
            'values': [0.0005, 0.005, 0.05]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="allay-ds-23")

def train(x_train_vect, y_train, x_val_vect, y_val):
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'epochs': 3,
        'weight_decay': 0.005,
        'learning_rate': 1e-3,
        'seed': 42
    }

    wandb.init(config=config_defaults)
    config = wandb.config
    model = Sequential()
    model.add(Dense(128, input_dim=2000, activation='relu', kernel_regularizer=regularizers.l2(config.weight_decay)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config.weight_decay)))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config.weight_decay)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(config.weight_decay)))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics=['accuracy'])

    model.fit(x_train_vect, y_train, batch_size=32,
                    steps_per_epoch=len(x_train_vect) / 32, epochs=config.epochs,
                    validation_data=(x_val_vect, y_val),
                    callbacks=[WandbCallback(validation_data=(x_val_vect, y_val), labels=["appropriate", "inappropriate"])])

    #wandb.log({'accuracy':accuracy})

wandb.agent(sweep_id, train(x_train_vect, y_train, x_val_vect, y_val))
