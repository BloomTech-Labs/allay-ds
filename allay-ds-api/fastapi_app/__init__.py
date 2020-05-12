"""Entry point for fastAPI application."""

from .app import create_app
# from .globals import NLP, MODEL

# Initialize FastAPI app
APP = create_app()
