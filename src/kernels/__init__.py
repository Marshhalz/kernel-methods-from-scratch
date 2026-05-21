"""
kernel-methods-from-scratch
============================
Kernel functions and RKHS theory implemented from mathematical first principles.

Reference:
    Shawe-Taylor & Cristianini (2004). Kernel Methods for Pattern Analysis.
    Cambridge University Press.
"""

from .kernels import LinearKernel, PolynomialKernel, RBFKernel

__all__ = ["LinearKernel", "PolynomialKernel", "RBFKernel"]
