"""Request and response item definitions."""

from typing import List

from pydantic import BaseModel, Field


class ReviewRequestItem(BaseModel):
    """Request body for /check_review endpoint"""
    text: str = Field(
        ...,
        title='Review Text',
        description='User submitted review content as a single string',
    )


class ReviewResponseItem(BaseModel):
    """Response body for /check_review endpoint"""
    text: str = Field(
        ...,
        title='Review Text',
        description='User submitted review content as a single string',
    )
    flag: int = Field(
        ...,
        title='Ok/Review/Block flag',
        description='0=OK, 1=REVIEW, 2=BLOCK',
        ge=0,
        le=2
    )
    score: float = Field(
        ...,
        title='Inappropriateness Score',
        description='Probability of inappropriateness of review content',
        ge=0.0,
        le=1.0,
    )
