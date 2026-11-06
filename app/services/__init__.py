"""Model back-end adapters for the platform.

Each adapter wraps one module behind the same small interface (``ModelAdapter``). The web layer
never imports torch / plink2 directly, so the app runs (and is testable) without them. An adapter
whose model is missing returns results with ``available=False`` and an explanation. It never
returns placeholder numbers.
"""
from app.services.base import ModelAdapter
from app.services.mammogram import MammogramClassifierAdapter
from app.services.mirai import MiraiAdapter
from app.services.prs import PRSAdapter

__all__ = ["ModelAdapter", "MammogramClassifierAdapter", "MiraiAdapter", "PRSAdapter"]
