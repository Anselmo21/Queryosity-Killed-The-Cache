# Deep-Q Network

See <https://arxiv.org/abs/2007.10568>

## Dependencies

The minimal set of dependencies for Python 3.9+ is found in [environment.yml](./environment.yml).
For exact reproducibility with my Python 3.13 environment, see [environment-full.yml](./environment-full.yml).

## Training and Generating Schedule

Both [dqn.ipynb](./dqn.ipynb) and [dqn.notebook.py](./dqn.notebook.py) contain code for training the DQN, saving the PyTorch model, and generating a schedule from it. You need to specify:
- a name for the model,
- the cache capacity (in 8kb pages).

Both the `.ipynb` and `.py` have the same content, just reproduced for preference of notebook vs script.

## Converting Between Notebook and Python File

To convert between Jupyter notebook and Python script, I use the [Jupytext](https://jupytext.readthedocs.io/en/latest/) package. It can be installed via Pip:

```bash
pip install jupytext
```

### Conversion from Notebook to Python Script

```bash
jupytext --to py:percent notebook.ipynb -o notebook.py
```

### Conversion from Python Script to Notebook:

```bash
jupytext --to ipynb notebook.py -o notebook.ipynb
```

### Clearing Notebook Outputs

Also, for clearing notebook outputs, I use the [nbstripout](https://github.com/kynan/nbstripout) command:

```bash
nbstripout notebook.ipynb
```
