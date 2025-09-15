# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:29:04 2025

@author: YAKE
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import yaml
import mujoco

def load_config(yaml_path: str) -> dict:
    """Load YAML config as dict."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_vec(x: float | Sequence[float], n: int) -> np.ndarray:
    """Return length-n vector from scalar or validate shape (n,)."""
    if np.isscalar(x):
        return np.full(n, float(x))
    x = np.asarray(x, dtype=float)
    assert x.shape == (n,), f"expected shape ({n},), got {x.shape}"
    return x


def to_mat3(K: float | np.ndarray) -> np.ndarray:
    """Return 3x3 matrix from scalar or validate (3,3)."""
    if np.isscalar(K):
        return np.eye(3, dtype=float) * float(K)
    K = np.asarray(K, dtype=float)
    assert K.shape == (3, 3), "expect scalar or (3,3) matrix"
    return K


# -----------------------------
# SO(3) utilities
# -----------------------------
def _project_to_SO3(R: np.ndarray) -> np.ndarray:
    """Project a matrix to the closest rotation (proper orthogonal)."""
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj


def rotmat_to_rotvec(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Log map: SO(3) -> so(3) (rotation vector). Robust near 0 and pi."""
    R = _project_to_SO3(np.asarray(R, float))
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(c))

    # near zero
    if theta < 1e-8:
        w_hat = 0.5 * (R - R.T)
        return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

    # near pi (handle axis extraction stably)
    if np.pi - theta < 1e-6:
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            x = axis[0]; y = A[0, 1] / (x + eps); z = A[0, 2] / (x + eps)
            v = np.array([x, y, z])
        elif axis[1] >= axis[0] and axis[1] >= axis[2]:
            y = axis[1]; x = A[0, 1] / (y + eps); z = A[1, 2] / (y + eps)
            v = np.array([x, y, z])
        else:
            z = axis[2]; x = A[0, 2] / (z + eps); y = A[1, 2] / (z + eps)
            v = np.array([x, y, z])
        v = v / (np.linalg.norm(v) + eps)
        return theta * v

    # generic
    w_hat = 0.5 * (R - R.T)
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return (theta / (np.sin(theta) + eps)) * w


def rotvec_to_rotmat(w: np.ndarray) -> np.ndarray:
    """Exp map: so(3) -> SO(3)."""
    w = np.asarray(w, float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    s = np.sin(theta); c = np.cos(theta)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return _project_to_SO3(R)


def so3_log(R: np.ndarray) -> np.ndarray:
    return rotmat_to_rotvec(R)


def so3_exp(w: np.ndarray) -> np.ndarray:
    return rotvec_to_rotmat(w)


def slerp_R(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Slerp between two rotation matrices."""
    R0 = _project_to_SO3(R0); R1 = _project_to_SO3(R1)
    Rrel = R0.T @ R1
    w = so3_log(Rrel)
    return R0 @ so3_exp(alpha * w)


# -----------------------------
# Reference interpolation
# -----------------------------
def sample_ref(arr: Optional[np.ndarray],
               t: float,
               ref_dt: float,
               t_total: float) -> Optional[np.ndarray]:
    """Linear interpolation of reference sequence (T, ...) at time t."""
    if arr is None:
        return None
    if t <= 0.0:
        return arr[0]
    if t >= t_total:
        return arr[-1]
    u = t / ref_dt
    i0 = int(np.floor(u))
    a = u - i0
    return (1 - a) * arr[i0] + a * arr[i0 + 1]


def sample_rot(Rseq: Optional[np.ndarray],
               t: float,
               ref_dt: float,
               t_total: float) -> Optional[np.ndarray]:
    """Time-interpolate rotation sequence (T, S, 3, 3) via per-index slerp."""
    if Rseq is None:
        return None
    if t <= 0.0:
        return Rseq[0]
    if t >= t_total:
        return Rseq[-1]
    u = t / ref_dt
    i0 = int(np.floor(u))
    a = u - i0
    R0 = Rseq[i0]       # (S,3,3)
    R1 = Rseq[i0 + 1]   # (S,3,3)
    S = R0.shape[0]
    Rout = np.empty_like(R0)
    for s in range(S):
        Rout[s] = slerp_R(R0[s], R1[s], a)
    return Rout


# -----------------------------
# Numerical derivatives on refs
# -----------------------------
def compute_qvel_qacc(model: mujoco.MjModel,
                      qpos_ref: np.ndarray,
                      dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unwrap hinge joints and compute numerical derivatives for qvel/qacc.

    Args:
        model: MuJoCo model (for joint meta).
        qpos_ref: (T, nq) reference positions.
        dt: reference sampling interval.

    Returns:
        qvel_ref: (T, nq)
        qacc_ref: (T, nq)
    """
    qpos_proc = np.asarray(qpos_ref, float).copy()

    # unwrap all hinge joints for smooth derivatives
    hinge_cols = [model.jnt_qposadr[j]
                  for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if hinge_cols:
        qpos_proc[:, hinge_cols] = np.unwrap(qpos_proc[:, hinge_cols], axis=0)

    qvel_ref = np.gradient(qpos_proc, dt, axis=0, edge_order=2)
    qacc_ref = np.gradient(qvel_ref, dt, axis=0, edge_order=2)
    return qvel_ref, qacc_ref


__all__ = [
    "load_config",
    "as_vec", "to_mat3",
    "rotmat_to_rotvec", "rotvec_to_rotmat", "so3_log", "so3_exp",
    "slerp_R",
    "sample_ref", "sample_rot",
    "compute_qvel_qacc",
]