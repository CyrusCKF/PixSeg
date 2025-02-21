import sys
from pathlib import Path

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.datasets import DATASET_METADATA, DATASET_ZOO

# TODO find ways to unit test dataset


def test_registry():
    assert len(DATASET_ZOO) >= 0
    assert len(DATASET_METADATA) >= 0
    assert DATASET_ZOO.keys() == DATASET_METADATA.keys()


if __name__ == "__main__":
    print(DATASET_ZOO)
    print(DATASET_METADATA)
