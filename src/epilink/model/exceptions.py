from __future__ import annotations


class EpiLinkError(Exception):
    """Base exception for all EpiLink-related errors."""


class ScenarioError(EpiLinkError, ValueError):
    """Raised when an invalid scenario is provided or parsed."""


class ConfigurationError(EpiLinkError, ValueError):
    """Raised when the EpiLink model is misconfigured."""


class SimulationError(EpiLinkError, RuntimeError):
    """Raised when an error occurs during simulation or draw generation."""
