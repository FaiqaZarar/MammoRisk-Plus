"""Validation labels: every output carries the populations and metric it was validated on.

A ``LabelledResult`` cannot be built without a known ``ValidationLabel``. Templates and the PDF
report only accept ``LabelledResult`` objects, so an unlabelled number can never reach a user.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATUSES = {"not_yet_validated", "validated", "exploratory"}
STATUS_TEXT = {
    "not_yet_validated": "Not yet validated",
    "validated": "Validated",
    "exploratory": "Exploratory (few events or small sample)",
}


class MissingLabelError(ValueError):
    """Raised when a result has no (or an unknown) validation label."""


@dataclass(frozen=True)
class ValidationLabel:
    key: str
    output: str
    module: str
    datasets: list[str]
    populations: list[str]
    not_validated_on: list[str]
    metric: str
    status: str
    value: float | None = None
    ci: list[float] | None = None
    source: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"label {self.key}: unknown status {self.status!r}")
        if self.status == "validated" and (self.value is None or not self.source):
            # A "validated" label without a number and a traceable source is not allowed.
            raise ValueError(f"label {self.key}: 'validated' needs value and source")

    @property
    def status_text(self) -> str:
        return STATUS_TEXT[self.status]

    @property
    def performance_text(self) -> str:
        """Human-readable performance; never invents a number."""
        if self.value is None:
            return f"{self.metric}: {self.status_text}"
        ci = f" (95% CI {self.ci[0]:.2f}-{self.ci[1]:.2f})" if self.ci else ""
        return f"{self.metric}: {self.value:.2f}{ci}"

    def lines(self) -> list[str]:
        """Plain-text lines used by the PDF report."""
        return [
            f"Validation status: {self.status_text}",
            f"Performance: {self.performance_text}",
            f"Validated on datasets: {', '.join(self.datasets)}",
            f"Populations: {', '.join(self.populations)}",
            f"NOT validated on: {', '.join(self.not_validated_on)}",
        ] + ([f"Source: {self.source}"] if self.source else [])


def load_labels(path: str | Path) -> dict[str, ValidationLabel]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {key: ValidationLabel(key=key, **entry) for key, entry in raw.items()}


@dataclass
class LabelledResult:
    """One output of one module, always paired with its validation label.

    ``available=False`` means the model/back-end is not installed; ``message`` explains why and
    no numbers are shown.
    """
    label: ValidationLabel
    title: str
    available: bool
    values: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    image_data_uri: str | None = None  # e.g. heatmap PNG as data: URI

    def __post_init__(self):
        if not isinstance(self.label, ValidationLabel):
            raise MissingLabelError(f"result {self.title!r} has no validation label")
        if not self.available and self.values:
            raise ValueError("an unavailable result must not carry values")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = asdict(self.label)
        d.pop("image_data_uri")  # too large for the session cookie
        return d

    @classmethod
    def from_dict(cls, d: dict, labels: dict[str, ValidationLabel]) -> "LabelledResult":
        key = (d.get("label") or {}).get("key")
        if key not in labels:
            raise MissingLabelError(f"unknown label {key!r}")
        return cls(label=labels[key], title=d["title"], available=d["available"],
                   values=d.get("values", {}), message=d.get("message", ""))


def make_result(labels: dict[str, ValidationLabel], key: str, **kwargs) -> LabelledResult:
    if key not in labels:
        raise MissingLabelError(f"no validation label registered for {key!r}")
    return LabelledResult(label=labels[key], **kwargs)
