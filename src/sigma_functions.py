"""
Sigma functions for DSGRN regulatory networks.

Dual-backend implementations (numpy for data generation, torch for PINN training)
of the Hill and ramp approximations to step-like sigma functions.

sigma_plus  (activation): increases from L to U as z increases
sigma_minus (repression): decreases from U to L as z increases
"""

import numpy as np
import torch

EPS = 1e-12


# ---------------------------------------------------------------------------
# Numpy implementations (for data generation / solve_ivp)
# ---------------------------------------------------------------------------

def sigma_plus_hill_np(z, T, L, U, d):
    """Hill activation: L + (U - L) * z^d / (T^d + z^d)."""
    z_safe = np.maximum(np.asarray(z, dtype=np.float64), EPS)
    zd = z_safe ** d
    Td = T ** d
    return L + (U - L) * zd / (Td + zd)


def sigma_minus_hill_np(z, T, L, U, d):
    """Hill repression: U + (L - U) * z^d / (T^d + z^d)."""
    z_safe = np.maximum(np.asarray(z, dtype=np.float64), EPS)
    zd = z_safe ** d
    Td = T ** d
    return U + (L - U) * zd / (Td + zd)


def sigma_plus_ramp_np(z, T, L, U, h):
    """Piecewise linear activation: L -> U around T with half-width h."""
    z = np.asarray(z, dtype=np.float64)
    left = T - h
    right = T + h
    frac = (z - left) / (2.0 * h + EPS)
    linear = L + (U - L) * frac
    return np.where(z <= left, L, np.where(z >= right, U, linear))


def sigma_minus_ramp_np(z, T, L, U, h):
    """Piecewise linear repression: U -> L around T with half-width h."""
    z = np.asarray(z, dtype=np.float64)
    left = T - h
    right = T + h
    frac = (z - left) / (2.0 * h + EPS)
    linear = U + (L - U) * frac
    return np.where(z <= left, U, np.where(z >= right, L, linear))


# ---------------------------------------------------------------------------
# Torch implementations (for PINN training, supports autograd)
# ---------------------------------------------------------------------------

def sigma_plus_hill_torch(z, T, L, U, d):
    """Hill activation (torch): L + (U - L) * z^d / (T^d + z^d)."""
    z_safe = torch.clamp(z, min=EPS)
    zd = z_safe ** d
    Td = T ** d
    return L + (U - L) * zd / (Td + zd)


def sigma_minus_hill_torch(z, T, L, U, d):
    """Hill repression (torch): U + (L - U) * z^d / (T^d + z^d)."""
    z_safe = torch.clamp(z, min=EPS)
    zd = z_safe ** d
    Td = T ** d
    return U + (L - U) * zd / (Td + zd)


def sigma_plus_ramp_torch(z, T, L, U, h):
    """Piecewise linear activation (torch): L -> U around T."""
    left = T - h
    right = T + h
    frac = (z - left) / (2.0 * h + EPS)
    linear = L + (U - L) * frac
    return torch.where(z <= left, L * torch.ones_like(z),
                       torch.where(z >= right, U * torch.ones_like(z), linear))


def sigma_minus_ramp_torch(z, T, L, U, h):
    """Piecewise linear repression (torch): U -> L around T."""
    left = T - h
    right = T + h
    frac = (z - left) / (2.0 * h + EPS)
    linear = U + (L - U) * frac
    return torch.where(z <= left, U * torch.ones_like(z),
                       torch.where(z >= right, L * torch.ones_like(z), linear))
