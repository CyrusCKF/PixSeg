import sys
from pathlib import Path

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.learn import CLASS_WEIGHTINGS, CRITERION_ZOO, LR_SCHEDULER_ZOO, OPTIMIZER_ZOO


def test_registry():
    assert len(CLASS_WEIGHTINGS) >= 0
    assert len(CRITERION_ZOO) >= 0
    assert len(LR_SCHEDULER_ZOO) >= 0
    assert len(OPTIMIZER_ZOO) >= 0
