"""Global variables to be initialized at launch.
"""

import os
import en_core_web_sm
from tensorflow.keras.models import load_model

# Initialize spaCy NLP model
NLP = en_core_web_sm.load()

# Load text classifier
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL = load_model(dir_path + '/pickles/cnn-model.h5')
