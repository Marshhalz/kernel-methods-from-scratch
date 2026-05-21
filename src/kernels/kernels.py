"""
Standard kernel functions.

Each kernel is derived from Shawe-Taylor & Cristianini (2004), Chapter 2.
All kernels extend ``Kernel`` and inherit ``matrix()`` and ``is_psd()``.

Implemented kernels
-------------------
LinearKernel       k(x, y) = x^T y
PolynomialKernel   k(x, y) = (x^T y + c)^d
RBFKernel          k(x, y) = exp(-||x - y||^2 / 2*sigma^2)
"""

from __future__ import annotations

import numpy as np

from .base import Kernel


class LinearKernel(Kernel):
    """Linear kernel: k(x, y) = <x, y> = x^T y.

    This is the simplest valid kernel.  Its implicit feature map is the
    identity: phi(x) = x, so the feature space IS the input space.

    PSD proof: the Gram matrix K = X X^T is PSD by construction for any X,
    since z^T K z = ||X^T z||^2 >= 0 for all z.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(x, y))

    def _params(self) -> dict:
        return {}


class PolynomialKernel(Kernel):
    """Polynomial kernel: k(x, y) = (x^T y + c)^d.

    Implicitly maps inputs to a feature space containing all monomials of
    degree up to d.  For d=2 in R^2, the feature map is:
        phi(x) = [x1^2, x2^2, sqrt(2)*x1*x2, sqrt(2c)*x1, sqrt(2c)*x2, c]

    PSD proof: the linear kernel is PSD, and sums and products of PSD kernels
    are PSD (closure properties, Shawe-Taylor Proposition 2.18).  Adding the
    constant c is equivalent to appending a constant feature; raising to power
    d is a product of d PSD kernels.

    Parameters
    ----------
    degree : int
        Polynomial degree d >= 1.
    c : float
        Inhomogeneity constant c >= 0.  c=0 gives a homogeneous polynomial.
    """

    def __init__(self, degree: int = 3, c: float = 1.0) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if c < 0:
            raise ValueError(f"c must be >= 0, got {c}")
        self.degree = degree
        self.c = c

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((np.dot(x, y) + self.c) ** self.degree)

    def _params(self) -> dict:
        return {"degree": self.degree, "c": self.c}


class RBFKernel(Kernel):
    """Radial Basis Function (Gaussian) kernel.

    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Also written as k(x, y) = exp(-gamma * ||x - y||^2) with gamma = 1/(2*sigma^2).

    This kernel is *universal*: it can approximate any continuous function on a
    compact domain to arbitrary precision (Micchelli et al., 2006).  The feature
    space is infinite-dimensional — the Taylor expansion of the exponential
    gives monomials of all degrees.

    PSD proof: follows from Bochner's theorem.  The Gaussian is the Fourier
    transform of a positive measure (itself a Gaussian), so it is a valid
    positive definite function.

    An important property: k(x, x) = 1 for all x, meaning every point has
    unit norm in the implicit feature space.

    Parameters
    ----------
    sigma : float
        Bandwidth parameter sigma > 0.  Larger sigma = smoother kernel.
    """

    def __init__(self, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = sigma

    @property
    def gamma(self) -> float:
        """gamma = 1 / (2 * sigma^2) — alternative parameterisation."""
        return 1.0 / (2.0 * self.sigma ** 2)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.exp(-np.dot(diff, diff) * self.gamma))

    def _params(self) -> dict:
        return {"sigma": self.sigma}
