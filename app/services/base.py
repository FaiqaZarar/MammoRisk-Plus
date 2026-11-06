"""Common adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.labels import LabelledResult, ValidationLabel, make_result


class ModelAdapter(ABC):
    #: validation-label keys this adapter produces, in display order
    label_keys: tuple[str, ...] = ()
    name: str = "model"

    def __init__(self, labels: dict[str, ValidationLabel], config: dict):
        self.labels = labels
        self.config = config

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        """(is the back-end usable, human-readable reason if not)."""

    @abstractmethod
    def _predict(self, files: dict[str, Path]) -> list[LabelledResult]:
        """Run the real model. Only called when ``availability()`` is True."""

    def predict(self, files: dict[str, Path]) -> list[LabelledResult]:
        ok, reason = self.availability()
        if not ok:
            return self.unavailable(reason)
        return self._predict(files)

    def unavailable(self, reason: str) -> list[LabelledResult]:
        return [make_result(self.labels, key, title=self.labels[key].output, available=False,
                            message=f"Model not available: {reason}")
                for key in self.label_keys]


def repo_path(p: str | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
