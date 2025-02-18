import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from torch.utils.data import Dataset

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_zoo import DATASET_ZOO, METADATA_ZOO, DatasetEntry, DatasetMeta

T = TypeVar("T", bound=Dataset)
P = ParamSpec("P")


def register_dataset(
    train_kwargs: dict,
    val_kwargs: dict,
    name: str | None = None,
    meta_key: str | None = None,
    num_classes: int | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Can be used on functions or classes"""

    def wrapper(callable: Callable[P, T]) -> Callable[P, T]:
        key = callable.__name__ if name is None else name
        if key in DATASET_ZOO:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        final_key = meta_key
        if meta_key is None and num_classes is None:
            final_key = key
        DATASET_ZOO[key] = DatasetEntry(
            callable, train_kwargs, val_kwargs, final_key, num_classes
        )
        return callable

    return wrapper


# TODO better interface
def register_metadata(name: str, metadata: DatasetMeta):
    if name in METADATA_ZOO:
        raise ValueError(f"An entry is already registered under the name '{name}'.")
    METADATA_ZOO[name] = metadata


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str((Path(__file__) / "..").resolve()))
    import pytorch_datasets

    print(DATASET_ZOO)
