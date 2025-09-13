#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_tune_pid_spsa.py — Simple RL-style tuner (SPSA) for your feedback gains.

What it does
------------
- Loads your MuJoCo model and reference trajectories [qpos_hist.csv (required), xpos_hist.csv (optional)].
- Runs your existing control logic pattern:
  τ = τ_ff (inverse dynamics on qdd_d, gravity-off) + τ_fb (joint-space PD on e,de, with ramp)
      + J^T ( KX * ex + DX * (xdot_d - xdot) )
- Uses SPSA (a lightweight, derivative-free optimizer often used in RL) to tune 4 scalars:
    KP_TAU0, KD_TAU0   (joint torque PD gains)
    KX, DX             (task-space PD gains)
- Cost: time-average(qpos RMS error) + λ * time-average(|xpos error|) + λτ * mean(|τ|)
- Writes best params to JSON and prints them.

Run
---
  python rl_tune_pid_spsa.py --iters 60 --seconds 3.0
  python rl_tune_pid_spsa.py --iters 120 --seconds 6.0 --kx 800 --dx 50 --kp_tau 60 --kd_tau 2*sqrt(60)

Notes
-----
- Keep iterations small at first (30–60). If it improves, increase to 200+.
- If real-time viewer is heavy on your machine, it stays headless (no viewer); it uses mujoco.Renderer offscreen only when --render_mp4 is set.
- Requires: mujoco, numpy, (optional) imageio for mp4 export.

"""
from __future__ import annotations

from pathlib import Path
import json, math, argparse, time
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

TARGET_CANDIDATES_SITE = ("toes_r", "r_toe", "foot_r", "r_foot", "right_toe", "right_foot")
TARGET_CANDIDATES_BODY = ("calcn_r", "toes_r", "r_foot", "foot_r", "right_foot")


def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)

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

    # xref (optional)
    USE_XPOS_REF = False
    xpos_ref = None
    if XPOS_CSV.exists():
        xr = load_traj(XPOS_CSV)
        if xr.shape[0] == T and xr.shape[1] % 3 == 0:
            nb = xr.shape[1] // 3
            xpos_ref = xr.reshape(T, nb, 3)
            USE_XPOS_REF = True
        else:
            warn("xpos_hist.csv 维度与预期不符，忽略。")

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

        # find a foot/site to control in task space
        self.TARGET_IS_SITE = False
        self.TARGET_ID = -1
        for nm in TARGET_CANDIDATES_SITE:
            try:
                self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, nm)
                self.TARGET_IS_SITE = True
                info(f"task target: site '{nm}' (id={self.TARGET_ID})")
                break
            except Exception:
                pass
        if self.TARGET_ID < 0:
            for nm in TARGET_CANDIDATES_BODY:
                try:
                    self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                    self.TARGET_IS_SITE = False
                    info(f"task target: body '{nm}' (id={self.TARGET_ID})")
                    break
                except Exception:
                    pass
        if self.TARGET_ID < 0:
            warn("未找到合适的脚/末端对象，任务空间项将被跳过。")

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

        # 1) feedforward inverse dynamics on desired qdd_d (gravity off)

        self.data.qacc[:] = qdd_d
        mujoco.mj_inverse(self.model, self.data)
        tau_ff = self.data.qfrc_inverse.copy()
        # 2) torque-domain PD
        tau_fb = (Kp_tau * s) * e + (Kd_tau * s) * de
        tau = tau_ff + tau_fb

        # 3) task-space PD (if target found)
        ex_vec = None
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

            ex_vec = x_d - x
            f_task = KX * ex_vec + DX * (xdot_d - xdot)
            tau += Jpos.T @ f_task

        # apply
        self.data.qfrc_applied[:] = np.clip(tau, -TORQUE_LIM, TORQUE_LIM)
        mujoco.mj_step(self.model, self.data)

        # metrics
        eq_rms = float(np.linalg.norm(e) / math.sqrt(max(1, e.size)))
        ex_norm = float(np.linalg.norm(ex_vec)) if ex_vec is not None else 0.0
        mean_tau_abs = float(np.mean(np.abs(tau)))
        return eq_rms, ex_norm, mean_tau_abs

def rollout_cost(sim: ControllerSim, qpos_ref, qd_ref, qdd_ref, dt_ref,
                 gains, seconds: float, use_xref=False, xpos_ref=None,
                 w_q=1.0, w_x=0.5, w_tau=1e-3):
    """Run a short episode and return averaged cost."""
    T = qpos_ref.shape[0]
    sim.reset_state(qpos_ref[0], qd_ref[0])

    steps = int(round(seconds / sim.model.opt.timestep))
    t = 0.0
    sum_q = 0.0
    sum_x = 0.0
    sum_tau = 0.0
    n = 0

    for _ in range(steps):
        eq, ex, mtau = sim.step_once(t, qpos_ref, qd_ref, qdd_ref, dt_ref, gains,
                                     use_xref=use_xref, xpos_ref=xpos_ref)
        sum_q += eq
        sum_x += ex
        sum_tau += mtau
        n += 1
        t += sim.model.opt.timestep
        # stop if we run out of reference
        if t >= (T-1) * dt_ref:
            break

    if n == 0:
        return 1e9
    mq = sum_q / n
    mx = sum_x / n
    mt = sum_tau / n
    return w_q * mq + w_x * mx + w_tau * mt

# ======== SPSA optimizer ========
def spsa_optimize(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, xpos_ref, use_xref,
                  theta0, iters=60, seconds=3.0, bounds=None,
                  w_q=1.0, w_x=0.5, w_tau=1e-3, seed=0):
    """
    theta = [Kp_tau, Kd_tau, KX, DX]
    bounds is list of (lo, hi) for each dimension.
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)

    # SPSA hyperparams (tuned conservatively)
    a0 = 0.15     # initial step size
    c0 = 0.1      # initial perturb size
    A  = max(10, iters // 5)  # stability offset

    # helpful scalings to keep gains sane
    def project(x):
        if bounds is None:
            return x
        out = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        return out

    best = (float('inf'), theta.copy())

    for k in range(1, iters + 1):
        ak = a0 / (k + A) ** 0.602
        ck = c0 / (k) ** 0.101

        # Rademacher perturbation
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        thetap = project(theta + ck * delta)
        thetam = project(theta - ck * delta)

        Jp = rollout_cost(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, thetap, seconds, use_xref, xpos_ref, w_q, w_x, w_tau)
        Jm = rollout_cost(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, thetam, seconds, use_xref, xpos_ref, w_q, w_x, w_tau)

        ghat = (Jp - Jm) / (2.0 * ck * delta)
        theta = project(theta - ak * ghat)

        J = rollout_cost(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, theta, seconds, use_xref, xpos_ref, w_q, w_x, w_tau)
        if J < best[0]:
            best = (J, theta.copy())

        info(f"[{k:03d}/{iters}] Jp={Jp:.4e} Jm={Jm:.4e} |J|={J:.4e} theta={theta} best={best[0]:.4e}")

    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60, help="SPSA iterations")
    ap.add_argument("--seconds", type=float, default=3.0, help="episode length (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_xref", action="store_true", help="use xpos_hist.csv as task-space reference if available")
    ap.add_argument("--w_q", type=float, default=1.0, help="weight for qpos RMS error")
    ap.add_argument("--w_x", type=float, default=0.5, help="weight for xpos error norm")
    ap.add_argument("--w_tau", type=float, default=1e-3, help="weight for torque magnitude")

    # initial guesses (same ballpark as your script)
    ap.add_argument("--kp_tau", type=float, default=60.0)
    ap.add_argument("--kd_tau", type=float, default=2.0 * math.sqrt(60.0))
    ap.add_argument("--kx", type=float, default=800.0)
    ap.add_argument("--dx", type=float, default=50.0)

    # bounds
    ap.add_argument("--kp_tau_lo", type=float, default=5.0)
    ap.add_argument("--kp_tau_hi", type=float, default=1200.0)
    ap.add_argument("--kd_tau_lo", type=float, default=0.1)
    ap.add_argument("--kd_tau_hi", type=float, default=200.0)
    ap.add_argument("--kx_lo", type=float, default=10.0)
    ap.add_argument("--kx_hi", type=float, default=8000.0)
    ap.add_argument("--dx_lo", type=float, default=0.1)
    ap.add_argument("--dx_hi", type=float, default=400.0)

    ap.add_argument("--render_mp4", type=str, default="", help="if set, render a short clip with best gains to this mp4 path")
    args = ap.parse_args()

    # Load refs
    qpos_ref, qd_ref, qdd_ref, dt_ref, USE_XPOS_REF, xpos_ref = prep_refs()
    if args.use_xref:
        USE_XPOS_REF = USE_XPOS_REF and (xpos_ref is not None)

    # make sim
    sim = ControllerSim(XML_PATH)
    if qpos_ref.shape[1] != sim.model.nq:
        raise ValueError(f"CSV 列数({qpos_ref.shape[1]}) 与模型 nq({sim.model.nq}) 不一致；请确认轨迹与XML匹配。")

    theta0 = [args.kp_tau, args.kd_tau, args.kx, args.dx]
    bounds = [(args.kp_tau_lo, args.kp_tau_hi),
              (args.kd_tau_lo, args.kd_tau_hi),
              (args.kx_lo, args.kx_hi),
              (args.dx_lo, args.dx_hi)]

    info(f"Start SPSA: iters={args.iters}, seconds={args.seconds}, theta0={theta0}")
    bestJ, bestTheta = spsa_optimize(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, xpos_ref, USE_XPOS_REF,
                                     theta0, iters=args.iters, seconds=args.seconds, bounds=bounds,
                                     w_q=args.w_q, w_x=args.w_x, w_tau=args.w_tau, seed=args.seed)
    info(f"Best cost = {bestJ:.6e}, best gains = {bestTheta}  (Kp_tau, Kd_tau, KX, DX)")

    # save
    out = {
        "best_cost": bestJ,
        "gains": {
            "Kp_tau": float(bestTheta[0]),
            "Kd_tau": float(bestTheta[1]),
            "KX": float(bestTheta[2]),
            "DX": float(bestTheta[3]),
        },
        "seconds": args.seconds,
        "iters": args.iters,
        "weights": {"w_q": args.w_q, "w_x": args.w_x, "w_tau": args.w_tau},
        "bounds": {"kp_tau": [bounds[0][0], bounds[0][1]],
                   "kd_tau": [bounds[1][0], bounds[1][1]],
                   "kx": [bounds[2][0], bounds[2][1]],
                   "dx": [bounds[3][0], bounds[3][1]]},
    }
    json_path = ROOT / "best_gains_spsa.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    info(f"Saved best gains → {json_path}")

    # optional short render with the best params
    if args.render_mp4:
        try:
            import imageio
            path = Path(args.render_mp4)
            renderer = mujoco.Renderer(sim.model, width=960, height=540)
            fps = 60
            frames = int(round(args.seconds * fps))
            t = 0.0
            # reset to start and roll with best gains
            sim.reset_state(qpos_ref[0], qd_ref[0])
            with imageio.get_writer(path, fps=fps) as w:
                for _ in range(frames):
                    target_t = min(t + 1.0/fps, (qpos_ref.shape[0]-1)*dt_ref)
                    while t < target_t:
                        sim.step_once(t, qpos_ref, qd_ref, qdd_ref, dt_ref, bestTheta,
                                      use_xref=USE_XPOS_REF, xpos_ref=xpos_ref)
                        t += sim.model.opt.timestep
                    renderer.update_scene(sim.data)
                    w.append_data(renderer.render())
            info(f"Rendered best policy clip → {path}")
        except Exception as e:
            warn(f"Render failed: {e}")

if __name__ == "__main__":
    main()
