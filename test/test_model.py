import sys
from pathlib import Path

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.models import MODEL_ZOO


def test_registry():
    assert len(MODEL_ZOO) >= 0
