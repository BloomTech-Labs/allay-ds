"""Global variables.

These are variables which take some time to initialize and occupy large
chunks of memory. They do not change after initializing. They are here so 
they can be accessed by any submodule of this package without being loaded
again.
"""

import os

import en_core_web_sm
from tensorflow.keras.models import load_model

# Initialize spaCy NLP model
NLP = en_core_web_sm.load()

# Load text classifier
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL = load_model(dir_path + '/pickles/cnn-model.h5')
