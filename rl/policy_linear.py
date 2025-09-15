"""
Linear-Gaussian policy with a linear value baseline (REINFORCE-style updates).

- pi(a|s) = Normal(mu, diag(std^2)), where mu = W s + b
- Value baseline: V(s) = v^T s + c, fitted by ridge regression each update
- Entropy regularization on the policy
- Optional learning of log-std (diagonal) via score-function gradient

Batch dict expected from rollout:
  {
    "s":   (N, state_dim)        # states
    "a":   (N, action_dim)       # actions taken
    "adv": (N,)                  # advantages (ret - V), will be standardized inside
    "ret": (N,)                  # returns, for value fit
    # Optional (if you precompute): "mu" (N, action_dim), "std" (action_dim,)
  }

Usage:
  pol = LinearGaussianPolicy(state_dim=10, action_dim=4, init_std=0.08, seed=0,
                             lr_pi=1e-2, lr_v=1e-2, ent_coef=1e-3, learn_std=False)
  a, logp, ent, mu, std = pol.act(s, rng)
  pol.update(batch)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import json


@dataclass
class PolicyConfig:
    lr_pi: float = 1e-2         # policy step size
    lr_v: float = 1e-2          # value baseline step size (only affects ridge reg strength)
    ent_coef: float = 1e-3      # entropy regularization coefficient
    init_std: float = 0.08      # initial action std (per-dim)
    learn_std: bool = False     # whether to update log_std by policy gradient
    std_min: float = 1e-4       # clamp for numerical stability
    std_max: float = 1e+2
    ridge_reg: float = 1e-4     # ridge for value regression
    adv_norm_eps: float = 1e-8  # epsilon for advantage normalization


class LinearGaussianPolicy:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 0, cfg: PolicyConfig | None = None):
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        self.state_dim = state_dim; self.action_dim = action_dim
        rng = np.random.default_rng(seed)
        
        self.W = rng.normal(scale=0.05, size=(action_dim, state_dim))  # (A,S)
        self.b = np.zeros(action_dim, dtype=float)                     # (A,)
        self.log_std = np.log(cfg.init_std) * np.ones(action_dim, dtype=float)  # (A,)
        
        self.v = np.zeros(state_dim, dtype=float)   # (S,)
        self.c = 0.0

    # ------------------- policy/value queries -------------------
    def act(self, s: np.ndarray, rng: np.random.Generator):
        """
        Sample a ~ N(mu, diag(std^2)), return (a, logp, ent, mu, std).
        s: (state_dim,)
        """
        mu = self.W @ s + self.b                        # (A,)
        std = np.clip(np.exp(self.log_std), self.cfg.std_min, self.cfg.std_max)
        noise = rng.normal(size=mu.shape)
        a = mu + std * noise
        logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
        ent = 0.5 * np.sum(np.log(2 * np.pi * np.e) + 2 * np.log(std))
        return a, float(logp), float(ent), mu, std

    def value(self, s: np.ndarray) -> float:
        return float(self.v @ s + self.c)

    # ------------------- training update -------------------
    def update(self, batch: dict):
        """
        REINFORCE-like update with a linear value baseline and entropy bonus.
        - Update mean params (W,b) by policy gradient using (adv standardized)
        - Fit V(s) by ridge regression on returns
        - Optionally update log_std by gradient ascent on entropy-regularized objective
        """
        S = np.asarray(batch["s"], dtype=float)      # (N,S)
        A = np.asarray(batch["a"], dtype=float)      # (N,A)
        ADV = np.asarray(batch["adv"], dtype=float)  # (N,)
        RET = np.asarray(batch["ret"], dtype=float)  # (N,)

        N, state_dim = S.shape
        action_dim = A.shape[1]
        assert self.W.shape == (action_dim, state_dim)
        
        adv = (ADV - ADV.mean()) / (ADV.std() + self.cfg.adv_norm_eps)  # (N,)
        MU = (S @ self.W.T) + self.b[None, :]                   # (N,A)
        STD = np.clip(np.exp(self.log_std), self.cfg.std_min, self.cfg.std_max)  # (A,)
        inv_var = 1.0 / (STD**2 + 1e-12)                        # (A,)
        
        # gradient of log pi wrt mu is: (A - MU) / STD^2
        diff = (A - MU) * inv_var[None, :]                      # (N,A)
        gW = (diff * adv[:, None]).T @ S                        # (A,S)
        gb = np.sum(diff * adv[:, None], axis=0)                # (A,)

        # --- optional: update log_std ---
        # gradient wrt log_std for a diagonal Gaussian:
        #   d/dlog_std logpi = -1 + ((a-mu)^2 / std^2)
        # averaged with advantages (optionally could include ent coef as separate term)
        if self.cfg.learn_std:
            g_logstd = np.mean((-1.0 + ((A - MU) ** 2) * inv_var[None, :]) * adv[:, None], axis=0)
            self.log_std += self.cfg.lr_pi * g_logstd
            self.log_std = np.log(np.clip(np.exp(self.log_std), self.cfg.std_min, self.cfg.std_max))

        # --- mean parameter step (policy) ---
        self.W += self.cfg.lr_pi * gW / max(1, N)
        self.b += self.cfg.lr_pi * gb / max(1, N)

        # --- fit value baseline by ridge regression on returns ---
        X = np.hstack([S, np.ones((N, 1))])                     # (N,S+1)
        y = RET.reshape(-1, 1)                                   # (N,1)
        reg = self.cfg.ridge_reg * np.eye(X.shape[1])            # (S+1,S+1)
        theta = np.linalg.pinv(X.T @ X + reg) @ (X.T @ y)        # (S+1,1)
        self.v = theta[:-1, 0]
        self.c = float(theta[-1, 0])

    # ------------------- persistence -------------------
    def state_dict(self) -> dict:
        return {
            "W": self.W, "b": self.b, "log_std": self.log_std,
            "v": self.v, "c": self.c,
            "cfg": vars(self.cfg),
        }

    @staticmethod
    def from_state_dict(state: dict, state_dim: int, action_dim: int) -> "LinearGaussianPolicy":
        cfg = PolicyConfig(**state.get("cfg", {}))
        pol = LinearGaussianPolicy(state_dim, action_dim, seed=0, cfg=cfg)
        pol.W = state["W"]; pol.b = state["b"]; pol.log_std = state["log_std"]
        pol.v = state["v"]; pol.c = float(state["c"])
        return pol

    def save_npz(self, path):
        meta = {
            "arch": "linear_gaussian",
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
        }
        np.savez(
            path,
            W=self.W.astype(np.float32),
            b=self.b.astype(np.float32),
            log_std=self.log_std.astype(np.float32),
            version=np.array(1, dtype=np.int32),
            meta=np.array(json.dumps(meta)),
        )

    @staticmethod
    def load_npz(path, state_dim=None, action_dim=None):
        z = np.load(path)
        W = z["W"]
        b = z["b"]
        log_std = z["log_std"]
        if state_dim is not None and W.shape[1] != state_dim:
            raise ValueError(f"W.shape[1]={W.shape[1]} != state_dim={state_dim}")
        if action_dim is not None and W.shape[0] != action_dim:
            raise ValueError(f"W.shape[0]={W.shape[0]} != action_dim={action_dim}")
        pi = LinearGaussianPolicy(state_dim=W.shape[1], action_dim=W.shape[0])
        pi.W = W
        pi.b = b
        pi.log_std = log_std
        return pi