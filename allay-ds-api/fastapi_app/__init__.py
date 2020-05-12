"""Entry point for fastAPI application."""


import en_core_web_md
from tensorflow.keras.models import load_model

from .app import create_app

# Initialize FastAPI app
APP = create_app()

# Initialize spaCy NLP model
NLP = en_core_web_md.load()

# Load text classifier
MODEL = load_model('model.h5')
