from __future__ import annotations

import math
from typing import Any


LABEL_ORDER = ["Still", "Gesture", "Drinking", "Toasting", "Nodding"]
ACTIVE_LABELS = [label for label in LABEL_ORDER if label != "Still"]

LABEL_TO_STATE = {label: index for index, label in enumerate(LABEL_ORDER)}
STATE_TO_NAME = {index: label for label, index in LABEL_TO_STATE.items()}

LABEL_PRIORITY = {
    "Still": 0,
    "Nodding": 1,
    "Gesture": 2,
    "Toasting": 3,
    "Drinking": 4,
}

LABEL_COLORS = {
    "Still": "#d9d9d9",
    "Gesture": "#f4a261",
    "Drinking": "#2a9d8f",
    "Toasting": "#577590",
    "Nodding": "#b56576",
}

FINE_CLASS_TO_COARSE_LABEL = {
    "still": "Still",
    "idle": "Still",
    "gesture": "Gesture",
    "gesturing": "Gesture",
    "gesture stroke": "Gesture",
    "gesture phrase": "Gesture",
    "hand switch": "Gesture",
    "drink": "Drinking",
    "drinking": "Drinking",
    "raise to mouth": "Drinking",
    "at mouth": "Drinking",
    "at mouth or drinking hold": "Drinking",
    "return from mouth": "Drinking",
    "raise to lips": "Drinking",
    "return from lips": "Drinking",
    "toast": "Toasting",
    "toasting": "Toasting",
    "nod": "Nodding",
    "nodding": "Nodding",
    "head nod": "Nodding",
}


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text.lower() if text else None


def coarse_label(value: Any) -> str | None:
    normalized = normalize_label(value)
    if normalized is None:
        return None
    if normalized in FINE_CLASS_TO_COARSE_LABEL:
        return FINE_CLASS_TO_COARSE_LABEL[normalized]
    title_label = str(value).strip().title()
    if title_label in LABEL_TO_STATE:
        return title_label
    return None


def require_label(value: Any) -> str:
    label = coarse_label(value)
    if label is None:
        raise ValueError(f"Unknown activity label: {value!r}")
    return label
