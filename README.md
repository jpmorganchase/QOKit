# QOKit
QOKit

# Quantum Optimization of Hard Problems

A joint project between Global Technology Applied Research of JPMorgan Chase, Argonne National Laboratory and Quantinuum 

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
pip install .
```

Installation can be verified by running tests using `pytest`.

### MaxCut

For MaxCut, the datasets in `assets/maxcut_datasets/` must be inflated

### Testing

pytest is used for tests. On machines with a very large number of cores, tests take a long time due to inefficiencies of some of the simulators being tested. Use `taskset -c 0-1 pytest` to run tests using only two cores.

### Questions

Please do not push directly to `main`. To contribute to `code/`, create a separate branch and push there, then open a pull request. `pytest` is used for tests.

### Disclaimer

All information contained in this GitHub repository is deemed Proprietary Information and is therefore subject to the terms of the Non-Disclosure Agreement (“NDA”) dated 14 November 2022, as amended. You should be aware that the information contained in this repository is subject to strict confidentiality requirements. If you have any questions relating to the terms of the NDA or the use of this repository, please contact <quantum.computing@jpmchase.com>. In particular, you should seek advice before granting access to any new participant, or sharing any content outside of this repository

### License
This project uses the [Apache License 2.0](LICENSE).

### Copyright
JP Morgan Chase & Co