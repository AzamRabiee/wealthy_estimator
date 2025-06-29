from typing import List, Optional
from dataclasses import dataclass

@dataclass
class WealthyProfile:
    name: str
    net_worth: int
    occupation: str
    similarity_score: float

@dataclass
class PredictionResponse:
    estimated_net_worth: int
    similar_profiles: List[WealthyProfile]
    confidence_score: float
    currency: str = "USD"

@dataclass
class ErrorResponse:
    error: str
    detail: Optional[str] = None 