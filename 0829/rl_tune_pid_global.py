#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_tune_pid_global.py — Global search wrapper around SPSA (multi-start, log-space).
"""
from __future__ import annotations
from pathlib import Path
import json, math, argparse, time, csv, sys
import numpy as np
import mujoco

# ======== Paths (change if needed) ========
ROOT = Path(__file__).resolve().parent
XML_PATH  = ROOT / "RKOB_simplified_upper_with_marker.xml"
QPOS_CSV  = ROOT / "qpos_hist.csv"
XPOS_CSV  = ROOT / "xpos_hist.csv"   # optional

# ======== Simulation + control defaults ========
FPS_TRAJ     = 60.0
TIMESTEP     = 5e-4
RAMP_T       = 0.3
TORQUE_LIM   = 1500.0
SMOOTH_DERIV = True
SMOOTH_ALPHA = 0.2
FF_MODE = "full"  # 'full' or 'no_g'

SITE_CANDS = ("toes_r", "r_toe", "foot_r", "r_foot", "right_toe", "right_foot")
BODY_CANDS = ("calcn_r", "toes_r", "r_foot", "foot_r", "right_foot")

def info(*a): print("[INFO]", *a, flush=True)
def warn(*a): print("[WARN]", *a, flush=True)

def load_traj(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim != 2:
        raise ValueError("CSV 应为二维 [T, N]")
    return arr

def ramp_factor(t, T=RAMP_T):
    return 1.0 if t >= T else (t / T)

def prep_refs():
    qpos_ref = load_traj(QPOS_CSV)
    T, nq = qpos_ref.shape
    dt_ref = 1.0 / FPS_TRAJ

    qd_ref  = np.gradient(qpos_ref, dt_ref, axis=0)
    qdd_ref = np.gradient(qd_ref,   dt_ref, axis=0)
    if SMOOTH_DERIV:
        for arr in (qd_ref, qdd_ref):
            for i in range(1, T):
                arr[i] = (1 - SMOOTH_ALPHA) * arr[i-1] + SMOOTH_ALPHA * arr[i]

    USE_XPOS_REF = False
    xpos_ref = None
    if XPOS_CSV.exists():
        xr = load_traj(XPOS_CSV)
        if xr.shape[0] == T and xr.shape[1] % 3 == 0:
            nb = xr.shape[1] // 3
            xpos_ref = xr.reshape(T, nb, 3)
            USE_XPOS_REF = True

    return qpos_ref, qd_ref, qdd_ref, dt_ref, USE_XPOS_REF, xpos_ref

def time_to_index(sim_t: float, dt_ref: float, T: int) -> int:
    return int(np.clip(np.floor(sim_t / dt_ref), 0, T - 1))

class ControllerSim:
    def __init__(self, xml_path: Path):
        assert xml_path.exists(), f"XML 不存在: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)
        self.data_fk = mujoco.MjData(self.model)

        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        self.model.opt.timestep   = TIMESTEP

        self.TARGET_IS_SITE = False
        self.TARGET_ID = -1
        for nm in SITE_CANDS:
            try:
                self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, nm)
                self.TARGET_IS_SITE = True
                info(f"task target: site '{nm}' (id={self.TARGET_ID})")
                break
            except Exception:
                pass
        if self.TARGET_ID < 0:
            for nm in BODY_CANDS:
                try:
                    self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                    self.TARGET_IS_SITE = False
                    info(f"task target: body '{nm}' (id={self.TARGET_ID})")
                    break
                except Exception:
                    pass

    def reset_state(self, q0, qd0):
        self.data.qpos[:] = q0
        self.data.qvel[:] = qd0
        mujoco.mj_forward(self.model, self.data)

    def step_once(self, sim_t, qpos_ref, qd_ref, qdd_ref, dt_ref, gains, use_xref=False, xpos_ref=None):
        Kp_tau, Kd_tau, KX, DX = gains  # scalars

        T = qpos_ref.shape[0]
        i = time_to_index(sim_t, dt_ref, T)
        q_d, qd_d, qdd_d = qpos_ref[i], qd_ref[i], qdd_ref[i]

        q, qd = self.data.qpos.copy(), self.data.qvel.copy()
        s = ramp_factor(sim_t)
        e, de = q_d - q, qd_d - qd

        # 1) feedforward inverse dynamics
        if FF_MODE == "no_g":
            g_backup = self.model.opt.gravity.copy()
            self.model.opt.gravity[:] = 0.0, 0.0, 0.0
            self.data.qacc[:] = qdd_d
            mujoco.mj_inverse(self.model, self.data)
            tau_ff = self.data.qfrc_inverse.copy()
            self.model.opt.gravity[:] = g_backup
        else:
            self.data.qacc[:] = qdd_d
            mujoco.mj_inverse(self.model, self.data)
            tau_ff = self.data.qfrc_inverse.copy()

        # 2) torque-domain PD
        tau_fb = (Kp_tau * s) * e + (Kd_tau * s) * de
        tau = tau_ff + tau_fb

        # 3) task-space PD (if target found)
        if self.TARGET_ID >= 0:
            Jpos = np.zeros((3, self.model.nv))
            if self.TARGET_IS_SITE:
                x = self.data.site_xpos[self.TARGET_ID]
                mujoco.mj_jacSite(self.model, self.data, Jpos, None, self.TARGET_ID)
            else:
                x = self.data.xpos[self.TARGET_ID]
                mujoco.mj_jacBody(self.model, self.data, Jpos, None, self.TARGET_ID)
            xdot = Jpos @ self.data.qvel

            if use_xref and (xpos_ref is not None) and (self.TARGET_ID < xpos_ref.shape[1]):
                x_d = xpos_ref[i, self.TARGET_ID]
                xdot_d = np.zeros(3)
            else:
                self.data_fk.qpos[:] = qpos_ref[i]
                self.data_fk.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data_fk)
                x_d = self.data_fk.site_xpos[self.TARGET_ID] if self.TARGET_IS_SITE else self.data_fk.xpos[self.TARGET_ID]
                xdot_d = np.zeros(3)

            f_task = KX * (x_d - x) + DX * (xdot_d - xdot)
            tau += Jpos.T @ f_task

        self.data.qfrc_applied[:] = np.clip(tau, -TORQUE_LIM, TORQUE_LIM)
        mujoco.mj_step(self.model, self.data)

        eq_rms = float(np.linalg.norm(e) / max(1.0, np.sqrt(e.size)))
        ex_norm = float(np.linalg.norm(x_d - x)) if self.TARGET_ID >= 0 else 0.0
        mean_tau_abs = float(np.mean(np.abs(tau)))
        return eq_rms, ex_norm, mean_tau_abs

def rollout_metrics(sim: ControllerSim, qpos_ref, qd_ref, qdd_ref, dt_ref,
                    gains, seconds: float, use_xref=False, xpos_ref=None):
    T = qpos_ref.shape[0]
    sim.reset_state(qpos_ref[0], qd_ref[0])
    steps = int(round(seconds / sim.model.opt.timestep))
    t = 0.0
    sum_q = sum_x = sum_tau = 0.0
    n = 0
    for _ in range(steps):
        eq, ex, mtau = sim.step_once(t, qpos_ref, qd_ref, qdd_ref, dt_ref, gains,
                                     use_xref=use_xref, xpos_ref=xpos_ref)
        sum_q += eq; sum_x += ex; sum_tau += mtau; n += 1
        t += sim.model.opt.timestep
        if t >= (T-1) * dt_ref:
            break
    if n == 0: return (1e9,1e9,1e9)
    return (sum_q/n, sum_x/n, sum_tau/n)

def cost_from_components(mq, mx, mt, w_q, w_x, w_tau):
    return w_q * mq + w_x * mx + w_tau * mt

def rollout_metrics_avg(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, gains, seconds, use_xref, xpos_ref, K):
    mq = mx = mt = 0.0
    for _ in range(max(1,K)):
        mqi, mxi, mti = rollout_metrics(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, gains, seconds, use_xref, xpos_ref)
        mq += mqi; mx += mxi; mt += mti
    K = max(1,K)
    return (mq/K, mx/K, mt/K)

# ======== SPSA (with exposed a0, c0) ========
def spsa(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, xpos_ref, use_xref,
         theta, iters, seconds, bounds, w_q, w_x, w_tau, seed, avg_k,
         a0=0.05, c0=0.02, A=None, verbose=True, adapt=True):
    rng = np.random.default_rng(seed)
    theta = np.array(theta, dtype=float)
    if A is None: A = max(10, iters//3)

    def project(x):
        return np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])

    best = (float('inf'), theta.copy())
    no_improve = 0

    for k in range(1, iters+1):
        ak = a0 / (k + A) ** 0.602
        ck = c0 / (k) ** 0.101

        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        thetap = project(theta + ck * delta)
        thetam = project(theta - ck * delta)

        mq_p, mx_p, mt_p = rollout_metrics_avg(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, thetap, seconds, use_xref, xpos_ref, avg_k)
        Jp = cost_from_components(mq_p, mx_p, mt_p, w_q, w_x, w_tau)
        mq_m, mx_m, mt_m = rollout_metrics_avg(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, thetam, seconds, use_xref, xpos_ref, avg_k)
        Jm = cost_from_components(mq_m, mx_m, mt_m, w_q, w_x, w_tau)

        ghat = (Jp - Jm) / (2.0 * ck * delta)
        theta = project(theta - ak * ghat)

        mq, mx, mt = rollout_metrics_avg(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, theta, seconds, use_xref, xpos_ref, avg_k)
        J = cost_from_components(mq, mx, mt, w_q, w_x, w_tau)
        if J < best[0]:
            best = (J, theta.copy())
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            info(f"SPSA[{k:03d}/{iters}] J={J:.4e} (best {best[0]:.4e}) θ={theta}")

        # simple adaptation: if plateau, shrink steps
        if adapt and no_improve >= max(5, iters//6):
            a0 *= 0.7; c0 *= 0.7; no_improve = 0
            if verbose: info(f"  ↳ plateau detected ⇒ shrink a0,c0 → {a0:.3g},{c0:.3g}")

    return best

# ======== Latin-hypercube-like sampling in log-space ========
def sample_in_bounds(rng, bounds, n, log_space=True):
    d = len(bounds)
    # stratified samples per dimension
    U = (np.arange(n) + rng.random(n)) / n  # [i+u]/n
    X = np.zeros((n, d))
    for j in range(d):
        rng.shuffle(U)
        lo, hi = bounds[j]
        if log_space and lo>0 and hi>0:
            lo_l, hi_l = math.log(lo), math.log(hi)
            X[:, j] = np.exp(lo_l + U*(hi_l - lo_l))
        else:
            X[:, j] = lo + U*(hi - lo)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters_short", type=int, default=20)
    ap.add_argument("--iters_long", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--avg_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=48, help="number of initial samples")
    ap.add_argument("--topk", type=int, default=4, help="how many seeds to refine")
    ap.add_argument("--use_xref", action="store_true")
    ap.add_argument("--w_q", type=float, default=1.0)
    ap.add_argument("--w_x", type=float, default=0.5)
    ap.add_argument("--w_tau", type=float, default=1e-4)
    ap.add_argument("--log_space", action="store_true", help="opt in log space for gains")

    # bounds
    ap.add_argument("--kp_tau_lo", type=float, default=20.0)
    ap.add_argument("--kp_tau_hi", type=float, default=200.0)
    ap.add_argument("--kd_tau_lo", type=float, default=5.0)
    ap.add_argument("--kd_tau_hi", type=float, default=40.0)
    ap.add_argument("--kx_lo", type=float, default=200.0)
    ap.add_argument("--kx_hi", type=float, default=2000.0)
    ap.add_argument("--dx_lo", type=float, default=10.0)
    ap.add_argument("--dx_hi", type=float, default=120.0)

    ap.add_argument("--out", type=str, default="best_gains_global.json")
    ap.add_argument("--hist_csv", type=str, default="global_history.csv")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    qpos_ref, qd_ref, qdd_ref, dt_ref, HAS_XREF, xpos_ref = prep_refs()
    use_xref = args.use_xref and HAS_XREF and (xpos_ref is not None)

    sim = ControllerSim(XML_PATH)
    if qpos_ref.shape[1] != sim.model.nq:
        raise ValueError(f"CSV 列数({qpos_ref.shape[1]}) 与模型 nq({sim.model.nq}) 不一致；请确认轨迹与XML匹配。")

    bounds = [(args.kp_tau_lo, args.kp_tau_hi),
              (args.kd_tau_lo, args.kd_tau_hi),
              (args.kx_lo, args.kx_hi),
              (args.dx_lo, args.dx_hi)]

    # 1) global sampling of initial seeds
    seeds = sample_in_bounds(rng, bounds, args.seeds, log_space=args.log_space)

    # evaluate quick score for ranking (short rollout, avg_k=2)
    scores = []
    for i, th in enumerate(seeds):
        mq, mx, mt = rollout_metrics_avg(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, th, args.seconds*0.5, use_xref, xpos_ref, max(1, args.avg_k-1))
        J = cost_from_components(mq, mx, mt, args.w_q, args.w_x, args.w_tau)
        scores.append((J, th))
        info(f"[seed {i+1:02d}/{args.seeds}] score J={J:.4e} θ={th}")
    scores.sort(key=lambda x: x[0])
    top = scores[:max(1, args.topk)]

    # 2) short SPSA refine for top-K, then pick best
    refined = []
    for rank, (J0, th0) in enumerate(top, 1):
        info(f"== Refine seed#{rank} start from J0={J0:.4e} θ0={th0}")
        bestJ, bestTh = spsa(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, xpos_ref, use_xref,
                             th0, args.iters_short, args.seconds, bounds,
                             args.w_q, args.w_x, args.w_tau, seed=args.seed+rank, avg_k=args.avg_k,
                             a0=0.05, c0=0.02, verbose=True, adapt=True)
        refined.append((bestJ, bestTh))
    refined.sort(key=lambda x: x[0])
    bestJ, bestTh = refined[0]

    # 3) long refinement from the best so far
    info(f"== Long refine from best so far J={bestJ:.4e} θ={bestTh}")
    bestJ, bestTh = spsa(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, xpos_ref, use_xref,
                         bestTh, args.iters_long, args.seconds, bounds,
                         args.w_q, args.w_x, args.w_tau, seed=args.seed+1234, avg_k=args.avg_k,
                         a0=0.04, c0=0.015, verbose=True, adapt=True)

    out = {
        "best_cost": float(bestJ),
        "gains": {"Kp_tau": float(bestTh[0]), "Kd_tau": float(bestTh[1]), "KX": float(bestTh[2]), "DX": float(bestTh[3])},
        "seconds": args.seconds,
        "iters_short": args.iters_short,
        "iters_long": args.iters_long,
        "seeds": args.seeds,
        "topk": args.topk,
        "avg_k": args.avg_k,
        "log_space": args.log_space,
        "bounds": {"kp_tau": [args.kp_tau_lo, args.kp_tau_hi],
                   "kd_tau": [args.kd_tau_lo, args.kd_tau_hi],
                   "kx": [args.kx_lo, args.kx_hi],
                   "dx": [args.dx_lo, args.dx_hi]}
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    info(f"Saved global best → {Path(args.out).resolve()}  gains={out['gains']}  J={bestJ:.4e}")

if __name__ == "__main__":
    main()
