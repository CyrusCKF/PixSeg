# Technical

For those who want to have a deeper look at this project

## Folder structure

- `src\semantic_segmentation_toolkit` - Main source code
  - `datasets` - Custom datasets and associated metadata(labels, background index, etc ...)
  - `models` - Custom models, model weights and backbones
  - `learn` - Custom loss functions, class weights, optimizers, and learning rate schedulers
  - `utils` - Smaller components and utility functions
  - `pipeline` - Integrate components for  training and evaluation
- `tasks` - Entry points to run this project
- `test` - Unit test

## Actions

### Run unit test with coverage

`python -m pytest --cov=src`

### Build as package

<https://packaging.python.org/en/latest/tutorials/packaging-projects/>

Upload to TestPyPI

*Note* Rename folder with underscore first

```bash
python -m pip install --upgrade build
python -m build
python -m pip install --upgrade twine
python -m twine upload --repository testpypi dist/* --verbose
```
