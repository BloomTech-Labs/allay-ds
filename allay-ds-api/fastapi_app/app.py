"""fastAPI application core logic."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI

# from .models.rate_random import get_score
from .models.rate_cnn import get_score
from .request_response_items import (RecommendReponseItem,
                                     RecommendRequestItem, ReviewRequestItem,
                                     ReviewResponseItem)

assert load_dotenv(), 'failed to initialize environment'


def score_to_flag(score: float):
    """Classify an inappropriateness probability into discreet categories.

    "param score: float (0.0 -> 1.0), required.

    Returns:
    int: 0 = OK, 1 = REVIEW, 2 = BLOCK
    """
    assert isinstance(score, float), f'score type ({type(score)}) not float.'
    assert score >= 0.0 and score <= 1.0, \
        f'Score ({score}) outside acceptable range (0->1).'
    if score < 0.2:
        return 0
    elif score < 0.95:
        return 1
    else:
        return 2


def create_app():
    """Create and configure a fastAPI application instance.
    Returns:
    fastAPI application instance.
    """
    app = FastAPI(title=__name__)

    @app.post('/check_review', name='check_review',
              response_model=ReviewResponseItem)
    async def post_check_review(item: ReviewRequestItem, ):
        score = get_score(item.comment)
        flag = score_to_flag(score)
        return {
            'comment': item.comment,
            'flag': flag,
            'score': score
        }

    @app.post('/recommend', name='recommend',
              response_model=RecommendReponseItem)
    async def post_recommend(item: RecommendRequestItem):
        return {
            'user_id': item.user_id,
            'post_ids': item.post_ids
        }

    return app
