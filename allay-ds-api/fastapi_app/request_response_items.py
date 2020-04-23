"""Request and response item definitions."""

from typing import List

from pydantic import BaseModel, Field


class RecommendRequestItem(BaseModel):
    """Request body for /recommend endpoint"""
    user_id: int = Field(
        ...,
        title='User ID',
        description='User ID that can be looked up in database'
    )
    post_ids: List[int] = Field(
        ...,
        title='List of post IDs',
        description='List of intger post IDs that can be looked ip in database'
    )


class RecommendReponseItem(BaseModel):
    """Response body for /recommend endpoint"""
    user_id: int = Field(
        ...,
        title='User ID',
        description='User ID that can be looked up in database'
    )
    post_ids: List[int] = Field(
        ...,
        title='Sorted list of post IDs',
        description='List of intger post IDs sorted be relevance for user_id'
    )


class ReviewRequestItem(BaseModel):
    """Request body for /check_review endpoint"""
    comment: str = Field(
        ...,
        title='Review Text',
        description='User submitted review content as a single string',
    )


class ReviewResponseItem(BaseModel):
    """Response body for /check_review endpoint"""
    comment: str = Field(
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
