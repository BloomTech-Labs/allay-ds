"""Entry point for fastAPI application."""

from .app import create_app

# Initialize FastAPI app
APP = create_app()
