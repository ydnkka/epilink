"""Scorer adapters: observations in, scores out; evaluation truth stays separate."""
from .registry import SCORERS, ScoringContext

__all__ = ["SCORERS", "ScoringContext"]
