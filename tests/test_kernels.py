"""
Tests for kernel functions.

Every valid kernel must satisfy:
  1. Symmetry:              k(x, y) == k(y, x)
  2. Positive semi-def.:    all eigenvalues of the Gram matrix >= 0
  3. Self-similarity:       k(x, x) >= 0  (follows from PSD, tested separately)

For the RBF kernel we additionally test the unit-norm property: k(x, x) = 1.
"""

import numpy as np
import pytest

from kernels import LinearKernel, PolynomialKernel, RBFKernel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
X_2D = RNG.standard_normal((20, 2))   # 20 points in R^2
X_10D = RNG.standard_normal((15, 10)) # 15 points in R^10


# ---------------------------------------------------------------------------
# Parametrised kernel list
# ---------------------------------------------------------------------------

KERNELS = [
    LinearKernel(),
    PolynomialKernel(degree=2, c=1.0),
    PolynomialKernel(degree=3, c=0.5),
    RBFKernel(sigma=0.5),
    RBFKernel(sigma=2.0),
]


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", KERNELS)
def test_symmetry_2d(kernel):
    """k(x, y) must equal k(y, x) for all x, y."""
    for i in range(len(X_2D)):
        for j in range(i + 1, len(X_2D)):
            assert kernel(X_2D[i], X_2D[j]) == pytest.approx(
                kernel(X_2D[j], X_2D[i]), rel=1e-10
            ), f"{kernel}: symmetry failed for i={i}, j={j}"


# ---------------------------------------------------------------------------
# Positive semi-definiteness (Gram matrix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", KERNELS)
def test_psd_2d(kernel):
    """Gram matrix on 2D data must be PSD."""
    assert kernel.is_psd(X_2D), f"{kernel}: Gram matrix is not PSD on 2D data"


@pytest.mark.parametrize("kernel", KERNELS)
def test_psd_10d(kernel):
    """Gram matrix on 10D data must be PSD."""
    assert kernel.is_psd(X_10D), f"{kernel}: Gram matrix is not PSD on 10D data"


# ---------------------------------------------------------------------------
# Gram matrix shape and symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", KERNELS)
def test_gram_matrix_shape(kernel):
    K = kernel.matrix(X_2D)
    assert K.shape == (len(X_2D), len(X_2D))


@pytest.mark.parametrize("kernel", KERNELS)
def test_gram_matrix_symmetric(kernel):
    K = kernel.matrix(X_2D)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Kernel-specific properties
# ---------------------------------------------------------------------------

def test_rbf_unit_norm():
    """RBF kernel satisfies k(x, x) = 1 for all x and all sigma."""
    kernel = RBFKernel(sigma=1.0)
    for x in X_2D:
        assert kernel(x, x) == pytest.approx(1.0, abs=1e-12)


def test_linear_dot_product():
    """Linear kernel equals the explicit dot product."""
    kernel = LinearKernel()
    x, y = X_2D[0], X_2D[1]
    assert kernel(x, y) == pytest.approx(float(np.dot(x, y)), rel=1e-12)


def test_polynomial_degree1_equals_linear_shifted():
    """PolynomialKernel(degree=1, c=0) must equal LinearKernel."""
    linear = LinearKernel()
    poly   = PolynomialKernel(degree=1, c=0.0)
    for x, y in zip(X_2D[:5], X_2D[1:6]):
        assert poly(x, y) == pytest.approx(linear(x, y), rel=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_polynomial_invalid_degree():
    with pytest.raises(ValueError, match="degree"):
        PolynomialKernel(degree=0)


def test_polynomial_invalid_c():
    with pytest.raises(ValueError, match="c must be"):
        PolynomialKernel(c=-1.0)


def test_rbf_invalid_sigma():
    with pytest.raises(ValueError, match="sigma"):
        RBFKernel(sigma=0.0)
