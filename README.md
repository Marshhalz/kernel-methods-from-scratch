# Kernel Methods from Scratch

> Implementing kernel functions, RKHS theory, and kernel-based algorithms from mathematical first principles — following the seminal texts by Shawe-Taylor & Cristianini and Schölkopf & Smola.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Motivation

Kernel methods provide a principled way to apply linear algorithms in
high-dimensional (or infinite-dimensional) feature spaces without ever
computing the feature map explicitly.  The key insight — the **kernel trick**
— is that all we need is the inner product between feature vectors:

$$k(x, y) = \langle \phi(x),\, \phi(y) \rangle_{\mathcal{H}}$$

This project builds that intuition from the ground up:  every algorithm is
implemented from scratch against the theory, with tests verifying the
mathematical properties (symmetry, positive semi-definiteness, etc.).

---

## Theory overview

### What is a valid kernel?

A function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a **positive
semi-definite (PSD) kernel** if, for any finite set
$\{x_1, \ldots, x_n\} \subset \mathcal{X}$, the **Gram matrix** $K$ defined by

$$K_{ij} = k(x_i, x_j)$$

is symmetric and positive semi-definite (all eigenvalues $\geq 0$).

By **Mercer's theorem**, every PSD kernel corresponds to an inner product in
some Hilbert space $\mathcal{H}$ — the **Reproducing Kernel Hilbert Space
(RKHS)** associated with $k$.

### Kernels implemented so far

| Kernel | Formula | Feature space |
|--------|---------|---------------|
| **Linear** | $k(x,y) = x^\top y$ | $\mathbb{R}^d$ (identity map) |
| **Polynomial** | $k(x,y) = (x^\top y + c)^d$ | All monomials up to degree $d$ |
| **RBF / Gaussian** | $k(x,y) = \exp\!\left(-\dfrac{\lVert x-y \rVert^2}{2\sigma^2}\right)$ | $\mathcal{H}$ (infinite-dimensional) |

### Key properties verified in tests

- **Symmetry** — $k(x, y) = k(y, x)$
- **PSD Gram matrix** — verified via eigenvalue decomposition
- **RBF unit-norm** — $k(x, x) = 1$ for all $x$
- **Consistency** — `PolynomialKernel(d=1, c=0)` $\equiv$ `LinearKernel`

---

## Project structure

```
kernel-methods-from-scratch/
├── src/kernels/
│   ├── base.py          # Abstract Kernel class + Gram matrix utilities
│   └── kernels.py       # LinearKernel, PolynomialKernel, RBFKernel
├── tests/
│   └── test_kernels.py  # PSD, symmetry, and property tests
├── notebooks/           # Exploratory notebooks (coming soon)
└── pyproject.toml
```

---

## Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Kernel functions (linear, polynomial, RBF) | ✅ Done |
| 2 | Kernel matrix visualisation + feature space intuition | 🔜 Next |
| 3 | Kernel PCA — dimensionality reduction in RKHS | ⬜ Planned |
| 4 | Kernel ridge regression | ⬜ Planned |
| 5 | Support Vector Machines (SVM) from scratch | ⬜ Planned |
| 6 | Gaussian Processes — the probabilistic view of kernels | ⬜ Planned |

---

## Quick start

```bash
# Clone and set up (requires Python 3.11+)
git clone https://github.com/<your-username>/kernel-methods-from-scratch.git
cd kernel-methods-from-scratch

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

```python
import numpy as np
from kernels import LinearKernel, PolynomialKernel, RBFKernel

X = np.random.randn(10, 3)

rbf = RBFKernel(sigma=1.0)

# Evaluate a single kernel value
print(rbf(X[0], X[1]))        # scalar

# Compute the full Gram matrix
K = rbf.matrix(X)             # shape (10, 10)

# Verify positive semi-definiteness
print(rbf.is_psd(X))          # True
```

Run the tests:

```bash
pytest
```

---

## References

- Shawe-Taylor, J. & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*. Cambridge University Press.
- Schölkopf, B. & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- Mercer, J. (1909). Functions of positive and negative type and their connection with the theory of integral equations. *Phil. Trans. R. Soc. London A*, 209, 415–446.

---

*Part of an ongoing series building ML algorithms from mathematical foundations.*
