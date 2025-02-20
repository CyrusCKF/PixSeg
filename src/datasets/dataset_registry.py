import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from torch.utils.data import Dataset

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_zoo import DATASET_METADATA, DATASET_ZOO, DatasetEntry, DatasetMeta

T = TypeVar("T", bound=Dataset)
P = ParamSpec("P")


def register_dataset(
    train_kwargs: dict,
    val_kwargs: dict,
    meta: str | int | DatasetMeta,
    name: str | None = None,
):
    """Can be used on functions or classes

    Args:
        meta: register metadata of this dataset based on types
            - `str` - refer to key of other metadata
            - `int` - number of classes, and other default values based on this
            - `DatasetMeta` - complete information
    """

    def wrapper(callable: Callable[P, T]) -> Callable[P, T]:
        key = callable.__name__ if name is None else name
        if key in DATASET_ZOO or key in DATASET_METADATA:
            raise KeyError(f"An entry is already registered under the key '{key}'.")

        DATASET_ZOO[key] = DatasetEntry(callable, train_kwargs, val_kwargs)

        meta_entry: DatasetMeta | str = (
            meta if not isinstance(meta, int) else DatasetMeta.default(meta)
        )
        DATASET_METADATA[key] = meta_entry
        return callable

    return wrapper


def resolve_metadata(key: str) -> DatasetMeta:
    try:
        meta_entry = DATASET_METADATA[key]
        if isinstance(meta_entry, DatasetMeta):
            return meta_entry
        return resolve_metadata(meta_entry)
    except KeyError:
        raise KeyError(f"Cannot resolve metadata for key {key}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str((Path(__file__) / "..").resolve()))
    import pytorch_datasets

    print(DATASET_ZOO.keys())
    print(DATASET_METADATA.keys())
    print(resolve_metadata("SBD"))
