from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScenarioScore:
    """Compatibility summary for one latent scenario."""

    time_percentile: float
    time_compatibility: float
    genetic_percentile: float
    genetic_compatibility: float
    compatibility: float

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def to_dict(self) -> dict[str, float]:
        return {
            "time_percentile": self.time_percentile,
            "time_compatibility": self.time_compatibility,
            "genetic_percentile": self.genetic_percentile,
            "genetic_compatibility": self.genetic_compatibility,
            "compatibility": self.compatibility,
        }


@dataclass(frozen=True, slots=True)
class PairCompatibilityResult:
    """Typed result object returned by :meth:`epilink.EpiLink.score_pair`."""

    target: str | tuple[str, ...]
    target_labels: tuple[str, ...]
    target_compatibility: float
    scenario_scores: dict[str, ScenarioScore]

    def __getitem__(self, key: str) -> Any:
        value = getattr(self, key)
        if key == "target_labels":
            return list(value)
        if key == "scenario_scores":
            return {label: score.to_dict() for label, score in value.items()}
        if key == "target" and isinstance(value, tuple):
            return list(value)
        return value

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self["target"],
            "target_labels": self["target_labels"],
            "target_compatibility": self.target_compatibility,
            "scenario_scores": self["scenario_scores"],
        }


__all__ = ["PairCompatibilityResult", "ScenarioScore"]
