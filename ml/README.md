# Deep-Q Network

See <https://arxiv.org/abs/2007.10568>

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
