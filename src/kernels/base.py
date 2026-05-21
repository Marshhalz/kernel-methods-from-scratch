"""
Abstract base class for kernel functions.

A kernel k : X x X -> R is a symmetric, positive semi-definite (PSD) function.
Equivalently, for any finite set {x_1, ..., x_n} subset X, the Gram matrix K
defined by K[i,j] = k(x_i, x_j) must be PSD (all eigenvalues >= 0).

By Mercer's theorem, every such kernel implicitly defines a feature map
phi : X -> H into some (possibly infinite-dimensional) Hilbert space H, such
that k(x, y) = <phi(x), phi(y)>_H.  We never need to compute phi explicitly —
this is the kernel trick.

Reference:
    Shawe-Taylor & Cristianini (2004), Chapter 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Kernel(ABC):
    """Abstract base class for kernel functions.

    Subclasses must implement ``__call__(x, y) -> float``.
    The ``matrix`` and ``is_psd`` methods are provided here and work for any
    subclass without modification.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate k(x, y).

        Parameters
        ----------
        x, y : np.ndarray
            Input vectors of the same shape.

        Returns
        -------
        float
            The scalar kernel value k(x, y).
        """

    def matrix(self, X: ArrayLike) -> np.ndarray:
        """Compute the n x n Gram (kernel) matrix for a dataset X.

        K[i, j] = k(X[i], X[j])

        Symmetry is exploited: only n*(n+1)/2 kernel evaluations are performed.

        Parameters
        ----------
        X : array-like of shape (n, d)
            Dataset of n points in d-dimensional input space.

        Returns
        -------
        np.ndarray of shape (n, n)
            Symmetric Gram matrix.
        """
        X = np.asarray(X)
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = self(X[i], X[j])
                K[i, j] = val
                K[j, i] = val  # exploit symmetry
        return K

    def is_psd(self, X: ArrayLike, tol: float = 1e-8) -> bool:
        """Check whether the Gram matrix on X is positive semi-definite.

        Uses the eigenvalue criterion: K is PSD iff all eigenvalues >= 0.
        ``np.linalg.eigvalsh`` is used because K is symmetric by construction,
        which guarantees real eigenvalues and is numerically more stable than
        the general ``eigvals``.

        Parameters
        ----------
        X : array-like of shape (n, d)
        tol : float
            Numerical tolerance for the zero-eigenvalue threshold.

        Returns
        -------
        bool
            True if all eigenvalues of K are >= -tol.
        """
        K = self.matrix(X)
        eigenvalues = np.linalg.eigvalsh(K)
        return bool(np.all(eigenvalues >= -tol))

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self._params().items())
        return f"{self.__class__.__name__}({params})"

    def _params(self) -> dict:
        """Return hyperparameter dict for __repr__. Override in subclasses."""
        return {}
