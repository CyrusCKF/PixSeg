# segmentic-segmentation-toolkit

 A simple semantic segmentation package using PyTorch

## Installation

1. Use python >= 3.10
1. Install PyTorch manually <https://pytorch.org/get-started/locally/>
1. Install dependencies in *pyproject.toml*
    - Run `pip install -e .[dev]`

## Packaging

<https://packaging.python.org/en/latest/tutorials/packaging-projects/>

Upload to TestPyPI

rename folder with underscore first

```bash
python -m pip install --upgrade build
python -m build
python -m pip install --upgrade twine
python -m twine upload --repository testpypi dist/* --verbose
```
