"""
kappa_limits
============
Theoretical performance bound estimation for binary classification using the
Crow & Watts information-theoretic framework.

Public API
----------
compute_theoretical_kappa(X, y, max_features, n_bins, n_iterations) -> dict
compute_actual_kappa(y_true, y_pred) -> float
"""

from .limits import compute_theoretical_kappa, compute_actual_kappa

__all__ = ["compute_theoretical_kappa", "compute_actual_kappa"]
