# Quantum Optimization Toolkit

![Tests](https://github.com/jpmorganchase/QOKit/actions/workflows/qokit-package.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2309.04841-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2309.04841)
[![PyPi version](https://badgen.net/pypi/v/qokit)](https://pypi.org/project/qokit/)
[![PyPI download month](https://img.shields.io/pypi/dm/qokit.svg)](https://pypi.org/project/qokit/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/qokit.svg)](https://pypi.org/project/qokit/)
[![PyPI license](https://img.shields.io/pypi/l/qokit.svg)](https://pypi.org/project/qokit/)

This repository contains fast CPU and GPU simulators for benchmarking the Quantum Approximate Optimization Algorithm, as well as scripts for generating matching quantum circuits for execution on hardware. See the [examples](./examples) folder for a demo of this package and check out the [blog post](https://www.jpmorgan.com/technology/technology-blog/quantum-optimization-research) describing the simulators.

### Install

Creating a virtual environment is recommended before installing.
```
python -m venv qokit
source qokit/bin/activate
pip install -U pip
```

Install requires `python>=3.9` and `pip >= 23`. It is recommended to update your pip using `pip install --upgrade pip` before install.

```
git clone https://github.com/jpmorganchase/QOKit.git
cd QOKit/
pip install -e .
```

Some optional parts of the package require additional dependencies. 
- Using commercial IP solvers to solve optimizations problems: `pip install -e .[solvers]`
- GPU simulation: `pip install -e .[GPU-CUDA12]`

Please note that the GPU dependency is specified for CUDA 12x. For other versions of CUDA, please follow cupy installation instructions.

If compilation fails, try installing just the Python version using `QOKIT_PYTHON_ONLY=1 pip install -e .`.

Installation can be verified by running tests using `pytest`.

#### MaxCut

For MaxCut, the datasets in `qokit/assets/maxcut_datasets` must be inflated

### Cite

For the simulators and other software tools, please cite
```
@inproceedings{Lykov2023,
  series = {SC-W 2023},
  title = {Fast Simulation of High-Depth QAOA Circuits},
  url = {http://dx.doi.org/10.1145/3624062.3624216},
  DOI = {10.1145/3624062.3624216},
  booktitle = {Proceedings of the SC ’23 Workshops of The International Conference on High Performance Computing,  Network,  Storage,  and Analysis},
  publisher = {ACM},
  author = {Lykov,  Danylo and Shaydulin,  Ruslan and Sun,  Yue and Alexeev,  Yuri and Pistoia,  Marco},
  year = {2023},
  month = nov,
  collection = {SC-W 2023}
}
```

For LABS data, please cite
```
@article{https://doi.org/10.48550/arxiv.2308.02342,
  doi = {10.48550/ARXIV.2308.02342},
  url = {https://arxiv.org/abs/2308.02342},
  author = {Shaydulin,  Ruslan and Li,  Changhao and Chakrabarti,  Shouvanik and DeCross,  Matthew and Herman,  Dylan and Kumar,  Niraj and Larson,  Jeffrey and Lykov,  Danylo and Minssen,  Pierre and Sun,  Yue and Alexeev,  Yuri and Dreiling,  Joan M. and Gaebler,  John P. and Gatterman,  Thomas M. and Gerber,  Justin A. and Gilmore,  Kevin and Gresh,  Dan and Hewitt,  Nathan and Horst,  Chandler V. and Hu,  Shaohan and Johansen,  Jacob and Matheny,  Mitchell and Mengle,  Tanner and Mills,  Michael and Moses,  Steven A. and Neyenhuis,  Brian and Siegfried,  Peter and Yalovetzky,  Romina and Pistoia,  Marco},
  keywords = {Quantum Physics (quant-ph),  Statistical Mechanics (cond-mat.stat-mech),  Emerging Technologies (cs.ET),  FOS: Physical sciences,  FOS: Physical sciences,  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem},
  howpublished = {Preprint at https://arxiv.org/abs/2308.02342},
}
```
