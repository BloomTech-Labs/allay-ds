"""fastAPI application core logic."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI

assert load_dotenv(), 'failed to initialize environment'


def create_app():
    """Create and configure a fastAPI application instance.
    Returns:
    fastAPI application instance.
    """
    app = FastAPI(title=__name__)

    @app.get('/')
    async def get_root():
        return {
            'message': 42,
            'useful': False
        }

    return app
