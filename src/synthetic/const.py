from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LABELS = ["negative", "neutral", "positive"]

LABELS_DISTRIBUTION = {
    "neutral": 0.5,
    "negative": 0.25,
    "positive": 0.25
}

# negative - 1300 - 0.25
# positive - 1400 - 0.25
# neutral - 3000 - 0.5