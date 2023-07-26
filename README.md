# Quantum Optimization of Hard Problems

![Tests](https://github.com/jpmorganchase/jpmc-argonne-quantum-optimization/actions/workflows/python-package.yml/badge.svg)

### Install

Creating a virtual environment is recommended before installing.
```
python -m venv qokit
source qokit/bin/activate
pip install -U pip
```

Install requires `pip >= 23`. It is recommended to update your pip using `pip install --upgrade pip` before install.

```
git clone https://github.com/jpmorganchase/jpmc-argonne-quantum-optimization.git
cd jpmc-argonne-quantum-optimization/
pip install -e .
```

If compilation fails, try installing just the Python version using `QOKIT_PYTHON_ONLY=1 pip install -e .`.

Installation can be verified by running tests using `pytest`.

### Usage

Check out [examples](./examples) folder for a demo of this package.

### MaxCut

For MaxCut, the datasets in `assets/maxcut_datasets/` must be inflated

