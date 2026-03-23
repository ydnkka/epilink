from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .genome import PackedGenomicData

T = TypeVar("T")
NDArrayInt8 = npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class SimulationSequenceSet(Generic[T]):
    """Paired deterministic and stochastic simulation outputs."""

    deterministic: T
    stochastic: T

    def __getitem__(self, key: str) -> T:
        if key not in {"deterministic", "stochastic"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("deterministic", "stochastic"))

    def __len__(self) -> int:
        return 2

    def __contains__(self, key: object) -> bool:
        return key in {"deterministic", "stochastic"}

    def to_dict(self) -> dict[str, T]:
        return {"deterministic": self.deterministic, "stochastic": self.stochastic}


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Typed result returned by :func:`simulate_genomic_sequences`."""

    packed: SimulationSequenceSet[PackedGenomicData]
    raw: SimulationSequenceSet[NDArrayInt8] | None

    def __getitem__(self, key: str) -> object:
        if key not in {"packed", "raw"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("packed", "raw"))

    def __len__(self) -> int:
        return 2

    def __contains__(self, key: object) -> bool:
        return key in {"packed", "raw"}

    def to_dict(self) -> dict[str, object]:
        return {
            "packed": self.packed.to_dict(),
            "raw": None if self.raw is None else self.raw.to_dict(),
        }


__all__ = ["SimulationResult", "SimulationSequenceSet"]
