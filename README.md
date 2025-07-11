## 🔥 CPU Enhancements (v0.3.x)

| ID | Component | What changed | Speed gain *vs v0.2* |
|----|-----------|--------------|----------------------|
| ①  | **Vectorised cost evaluation** | `get_configuration_cost_vector` turns the Python loop over 2^N bit-strings into one NumPy dot-product. | **≈ 20×** |
| ②  | **Numba brute-force JIT** | Parallel SIMD kernel for `2^N` energies (`brute_force_cost_vector`). | **≈ 5×** for N ≤ 20 |
| ③  | **Analytic gradient + L-BFGS-B** | Analytic ∇ shrinks optimiser calls; SciPy L-BFGS-B replaces BOBYQA. | **2–3×** fewer evaluations |
| ④  | **Phase-vector cache** | One-shot `exp()` of the diagonal phase, reused every QAOA layer. | **10–25 %** |

Combined (12 assets, p = 3) → **63 s → 23 s (× 2.7)** on laptop-CPU.

---

### 🧪 Reproduce on free Intel node (qBraid Lab)

```bash
# 0. Spin up a Large-CPU session (32 vCPU).
# 1. Import repo by tag (e.g. v0.3.2) – Lab builds env *qokit-po*.

conda activate qokit-po
cd QOKit
pip install -e .[optim,test]          # editable + SciPy/NLopt/Numba

# 2. Run unit tests – should be all green.
pytest -q

# 3. Baseline profile (loop cost + BOBYQA)
python scripts/benchmark_before_after.py \
       --Ns 16 17 18 19 20 21 22 23 24 25 \
       --ps 1 3 5 10 15 \
       --profile baseline 

# 4. Enhanced profile (vector+Numba+LBFGS+cache)
python scripts/benchmark_before_after.py \
       --Ns 16 17 18 19 20 21 22 23 24 25 \
       --ps 1 3 5 10 15 \
       --profile enhanced 
# └─ merges CSVs & writes:
#    results/before_after.csv
#    results/figure_before_after.png

# 5. Optional: print snapshot table for N = 16/20/25.
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/before_after.csv")
tbl = df.pivot(index="N", columns="p", values="percent_gain")
print(tbl.loc[[16,20,25], [1,3,5,10,15]].round(1).to_markdown())
PY


```bash
# clone and install (CPU-only)
git clone https://github.com/jerome79/QOKit.git
cd QOKit
pip install -e .[optim]          # pulls SciPy for L-BFGS-B

# run a 2-layer sweep on analytic-gradient path
python scripts/run_sweep.py 12 4 0.7 --p 1 2 --optim lbfgs

### Benchmark vs brute force

```bash
python scripts/benchmark_vs_bruteforce.py --Ns 16 20 24 --ps 1 5 10

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
- GPU simulation: `pip install -e .[GPU-CUDA12]`
- Generating LP files to solve LABS using commercial IP solvers (`qokit/classical_methods` and `examples/advanced/classical_solvers_for_LABS/`): `pip install -e .[solvers]`

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
