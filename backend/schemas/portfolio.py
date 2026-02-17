"""
Portfolio Schemas

Request and response models for the portfolio chat endpoint.
"""

from pydantic import BaseModel


class PortfolioChatRequest(BaseModel):
    message: str


class PortfolioChatResponse(BaseModel):
    response: str
