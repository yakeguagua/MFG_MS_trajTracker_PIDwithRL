#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_traj_rl.py — Online RL (REINFORCE) tuner for PD gains on top of your trajectory tracker.

- Removes constraint compensation.
- Keeps joint-space inverse dynamics (computed torque) and task-space (position + velocity only).
- RL outputs log-space deltas for 4 multipliers applied to your existing gains:
    alpha_p, alpha_d  → scale kp_q, kd_q vectors
    beta_x,  beta_v   → scale kx, dx scalars
- Training/eval closely follows rl_adaptive_pid_pg.py (REINFORCE with linear-Gaussian policy).

Usage:
  python track_traj_rl.py --xml RKOB_simplified_upper_with_marker.xml \
    --traj AJ026_Run_Comfortable.pkl --cfg cfg.yaml --seconds 6.0 --train_episodes 200
  python track_traj_rl.py --xml RKOB_simplified_upper_with_marker.xml \
    --traj AJ026_Run_Comfortable.pkl --cfg cfg.yaml --seconds 6.0 --eval

Outputs:
  - rl_gains_policy.npz
  - rl_eval_metrics.csv
  - rl_eval_summary.txt
"""
from pathlib import Path
import json, math, time, pickle, yaml, argparse, csv
import numpy as np
import mujoco

# ---------------------- Trajectory + Controller (modified from your track_traj.py) ----------------------
class TrajectoryTracker:
    REF_FPS = 50.0
    REF_DT  = 1.0 / REF_FPS

    def __init__(self, model, data, qpos_ref, config, xpos_ref=None):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv
        assert self.nq == self.nv
        self._scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        self.dt_sim = model.opt.timestep
        self.qpos_ref = np.asarray(qpos_ref, float)  # (T, nq)
        self.T = self.qpos_ref.shape[0]
        self.t_total = (self.T - 1) * self.REF_DT

        # --- config ---
        get = lambda k, d: config[k] if k in config else d
        self.task_enabled   = bool(get("task_enabled", True))
        self.task_mode      = get("task_mode", "body")  # 'body' or 'site'
        self.task_names     = list(get("task_names", []))

        # Base PD gains (joint space)
        self.base_kp_q = self._as_vec(get("kp_q", 60.0), self.nv)
        self.base_kd_q = self._as_vec(get("kd_q", 12.0), self.nv)

        # Base task gains (position/velocity only)
        self.base_kx = float(get("kx", 500.0))
        self.base_dx = float(get("dx", 50.0))

        # torque limits
        self.torque_limit = self._as_vec(get("torque_limit", 1500.0), self.nv)

        # references
        self.qvel_ref, self.qacc_ref = self._compute_qvel_qacc(self.qpos_ref, self.REF_DT)

        self.task_ids = self._resolve_task_ids(self.task_mode, self.task_names)
        self.xpos_ref, self.xvel_ref = self._build_task_refs(self.task_mode, self.task_ids, xpos_ref)

        # RL-scaled gains (initialize to 1.0 multipliers)
        self.alpha_p = 1.0  # scales base_kp_q
        self.alpha_d = 1.0  # scales base_kd_q
        self.beta_x  = 1.0  # scales base_kx
        self.beta_v  = 1.0  # scales base_dx

        # logs
        self._log_t = []
        self._log_mq = []
        self._log_mx = []
        self._log_mt = []
        self._last_tau = np.zeros(self.nv)

        # task target auto-discovery (first candidate found is used for metrics)
        self._metric_tid = self.task_ids[0] if len(self.task_ids) else -1
        self._target_is_site = (self.task_mode == 'site')

    @staticmethod
    def _as_vec(x, n):
        if np.isscalar(x): return np.full(n, float(x))
        x = np.asarray(x, float)
        assert x.shape == (n,), f"expected shape ({n},)"
        return x

    def _compute_qvel_qacc(self, qpos_ref, dt):
        qpos_proc = qpos_ref.copy()
        # unwrap hinge joints for smooth derivatives
        hinge_cols = [self.model.jnt_qposadr[j]
                      for j in range(self.model.njnt)
                      if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if hinge_cols:
            qpos_proc[:, hinge_cols] = np.unwrap(qpos_proc[:, hinge_cols], axis=0)
        qvel_ref = np.gradient(qpos_proc, dt, axis=0, edge_order=2)
        qacc_ref = np.gradient(qvel_ref, dt, axis=0, edge_order=2)
        return qvel_ref, qacc_ref

    def _resolve_task_ids(self, mode, names):
        if not self.task_enabled:
            return []
        ids = []
        if mode == 'body':
            all_names = [self.model.body(i).name for i in range(self.model.nbody)]
            for nm in names:
                assert nm in all_names, f"未知 body: {nm}"
                bid = all_names.index(nm)
                assert bid != 0, "不要选择 worldbody"
                ids.append(bid)
        elif mode == 'site':
            all_names = [self.model.site(i).name for i in range(self.model.nsite)]
            for nm in names:
                assert nm in all_names, f"未知 site: {nm}"
                ids.append(all_names.index(nm))
        else:
            raise ValueError("task_mode must be 'body' or 'site'")
        return ids

    def _fk_stack_over_traj(self, ids, *, is_body):
        d = self._scratch
        S = len(ids)
        X    = np.zeros((self.T, S, 3), dtype=float)
        Xdot = np.zeros_like(X)
        jacp = np.zeros((3, self.nv))
        for t in range(self.T):
            d.qpos[:] = self.qpos_ref[t]
            d.qvel[:] = self.qvel_ref[t]
            mujoco.mj_forward(self.model, d)
            for s, tid in enumerate(ids):
                if is_body:
                    X[t, s, :] = d.xpos[tid]
                    mujoco.mj_jacBody(self.model, d, jacp, None, tid)
                else:
                    X[t, s, :] = d.site_xpos[tid]
                    mujoco.mj_jacSite(self.model, d, jacp, None, tid)
                Xdot[t, s, :] = jacp @ d.qvel
        return X, Xdot

    def _build_task_refs(self, mode, ids, xpos_ref):
        if (not self.task_enabled) or len(ids) == 0:
            return None, None
        if xpos_ref is not None:
            assert mode == 'body', "提供 xpos_ref 时任务空间必须为 'body'"
            Xall = np.asarray(xpos_ref, float)
            assert Xall.ndim == 3 and Xall.shape[:2] == (self.T, self.model.nbody) and Xall.shape[2] == 3
            S = len(ids)
            Xsel = np.zeros((self.T, S, 3), dtype=float)
            for s, bid in enumerate(ids):
                assert bid != 0, "task_names/ids 不可包含 worldbody"
                Xsel[:, s, :] = Xall[:, bid, :]
            Xdot = np.zeros_like(Xsel)
            return Xsel, Xdot
        X, Xdot = self._fk_stack_over_traj(ids, is_body=(mode == 'body'))
        return X, Xdot

    def _sample_ref(self, arr, t):
        if arr is None: return None
        if t <= 0.0: return arr[0]
        if t >= self.t_total: return arr[-1]
        u = t / self.REF_DT
        i0 = int(np.floor(u)); a = u - i0
        return (1-a)*arr[i0] + a*arr[i0+1]

    # ---------------- 控制分量 ----------------
    def _tau_inverse(self, t):
        # computed torque: v = qdd_d + kd*de + kp*e, tau = ID(q, qd, v)
        q_d   = self._sample_ref(self.qpos_ref, t)
        qd_d  = self._sample_ref(self.qvel_ref, t)
        qdd_d = self._sample_ref(self.qacc_ref, t)
        q  = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        e  = q_d  - q
        de = qd_d - qd
        kp_q = self.alpha_p * self.base_kp_q
        kd_q = self.alpha_d * self.base_kd_q
        v  = qdd_d + kd_q * de + kp_q * e
        d = self._scratch
        d.qpos[:] = q
        d.qvel[:] = qd
        d.qacc[:] = v
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def _tau_task(self, t):
        if (not self.task_enabled) or (len(self.task_ids) == 0):
            return np.zeros(self.nv)
        x_d = self._sample_ref(self.xpos_ref, t)   # (S,3) or None
        v_d = self._sample_ref(self.xvel_ref, t)   # (S,3) or None
        tau_task = np.zeros(self.nv)
        jacp = np.zeros((3, self.nv))
        mujoco.mj_fwdPosition(self.model, self.data)
        qd = self.data.qvel
        Kx = self.beta_x * self.base_kx
        Dx = self.beta_v * self.base_dx
        for s, tid in enumerate(self.task_ids):
            if self.task_mode == 'body':
                x = self.data.xpos[tid].copy()
                mujoco.mj_jacBody(self.model, self.data, jacp, None, tid)
            else:
                x = self.data.site_xpos[tid].copy()
                mujoco.mj_jacSite(self.model, self.data, jacp, None, tid)
            v = jacp @ qd
            ex = (x_d[s] - x) if (x_d is not None) else (-x)
            ev = (v_d[s] - v) if (v_d is not None) else (-v)
            f_lin = Kx * ex + Dx * ev
            tau_task += jacp.T @ f_lin
        return tau_task

    def compute_tau(self, t):
        tau = self._tau_inverse(t) + self._tau_task(t)
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau

    def step_once(self, t):
        # one sim step; also return metrics used for RL
        q_d = self._sample_ref(self.qpos_ref, t)
        q   = self.data.qpos.copy()
        e   = q_d - q
        tau = self.compute_tau(t)
        self.data.qfrc_applied[:] = tau
        mujoco.mj_step(self.model, self.data)

        # metrics
        mq = float(np.linalg.norm(e) / math.sqrt(max(1, e.size)))      # joint RMS error
        mt = float(np.mean(np.abs(tau)))                                # mean |tau|
        # task error (if target exists)
        mx = 0.0
        if self.xpos_ref is not None and len(self.task_ids) > 0:
            x_d = self._sample_ref(self.xpos_ref, t)  # (S,3)
            if x_d is not None:
                X_now = []
                for tid in self.task_ids:
                    if self.task_mode == 'site':
                        X_now.append(self.data.site_xpos[tid])
                    else:
                        X_now.append(self.data.xpos[tid])
                X_now = np.asarray(X_now)             # (S,3)
                err_vec = x_d - X_now                 # (S,3)
                err_norm = np.linalg.norm(err_vec, axis=1)  # (S,)
                mx = float(err_norm.mean())
                # mx = float(np.sqrt((err_norm**2).mean()))
                # mx = float(err_norm.max()))
        else:
            mx = 0.0
        self._last_tau = tau.copy()
        self._log_t.append(float(t)); self._log_mq.append(mq); self._log_mx.append(mx); self._log_mt.append(mt)
        return mq, mx, mt

    def reset_to_start(self):
        self.data.qpos[:] = self.qpos_ref[0]
        self.data.qvel[:] = self.qvel_ref[0]
        mujoco.mj_forward(self.model, self.data)
        self._log_t.clear(); self._log_mq.clear(); self._log_mx.clear(); self._log_mt.clear()

# ---------------------- RL policy (Linear Gaussian, REINFORCE) ----------------------
class GainParam4:
    # log-space 4D params with per-step clipped delta
    def __init__(self, lo, hi):
        self.lo = np.array(lo, dtype=float)
        self.hi = np.array(hi, dtype=float)
        self.y_lo = np.log(self.lo); self.y_hi = np.log(self.hi)
        self.y = 0.5*(self.y_lo + self.y_hi)  # start at mid
    
    def step(self, dy, frac_step=0.05):
        max_step = frac_step * (self.y_hi - self.y_lo)
        dy = np.clip(dy, -max_step, max_step)
        self.y = np.clip(self.y + dy, self.y_lo, self.y_hi)
    
    def values(self):
        g = np.clip(np.exp(self.y), self.lo, self.hi)
        return g
    
    def normalized(self):
        return (self.y - self.y_lo) / (self.y_hi - self.y_lo + 1e-12)

class LinearGaussianPolicy:
    def __init__(self, state_dim, action_dim, init_std=0.08, seed=0, ent_coef=1e-3, lr=1e-2):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.05, size=(action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_std = np.log(init_std) * np.ones(action_dim)
        self.v = np.zeros(state_dim); self.c = 0.0
        self.lr_pi = lr; self.lr_v = lr; self.ent_coef = ent_coef
    
    def act(self, s, rng):
        mu = self.W @ s + self.b
        std = np.exp(self.log_std)
        a = mu + std * rng.normal(size=mu.shape)
        logp = -0.5*np.sum(((a-mu)/std)**2 + 2*np.log(std) + np.log(2*np.pi))
        ent  = 0.5*np.sum(np.log(2*np.pi*np.e) + 2*np.log(std))
        return a, float(logp), float(ent), mu, std
    
    def value(self, s):
        return float(self.v @ s + self.c)
    
    def update(self, batch):
        S = batch["s"]; A = batch["a"]; ADV = batch["adv"]
        MU = batch["mu"]; STD = batch["std"]
        N = S.shape[0]
        adv = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        inv_var = 1.0 / (STD**2 + 1e-12)
        gW = np.zeros_like(self.W); gb = np.zeros_like(self.b)
        for n in range(N):
            s = S[n]; diff = (A[n] - MU[n]) * inv_var
            gW += np.outer(diff * adv[n], s); gb += diff * adv[n]
        self.W += self.lr_pi * gW / N; self.b += self.lr_pi * gb / N
        X = np.hstack([S, np.ones((N,1))]); y = batch["ret"].reshape(-1,1)
        reg = 1e-4*np.eye(X.shape[1])
        theta = np.linalg.pinv(X.T@X + reg) @ (X.T @ y)
        self.v = theta[:-1,0]; self.c = theta[-1,0]

# ---------------------- Training/Eval wrappers ----------------------
def prep_refs_from_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)
    qpos_ref = np.asarray(traj["mj_qpos"], float)
    xpos_ref = traj.get("mj_xpos", None)
    return qpos_ref, xpos_ref

def rollout_episode(tracker: TrajectoryTracker, seconds, gp: GainParam4, policy: LinearGaussianPolicy, rng, action_stride: int = 1, frac_step: float = 0.05):
    # state features: [mq, mx, mt] normalized + time features + normalized gains (4)
    EQ_SCL = 0.2; EX_SCL = 0.03; MT_SCL = 500.0
    t = 0.0; k = 0
    S_list=[]; A_list=[]; R_list=[]; LOGP=[]; MU=[]; ENT=[]
    last_a = np.zeros(4, dtype=float)
    last_mu = np.zeros(4, dtype=float)
    last_std = np.exp(policy.log_std)
    while t < seconds and t < tracker.t_total:
        alpha_p, alpha_d, beta_x, beta_v = gp.values()
        tracker.alpha_p = alpha_p; tracker.alpha_d = alpha_d
        tracker.beta_x  = beta_x;  tracker.beta_v  = beta_v
        # step env
        mq, mx, mt = tracker.step_once(t)
        phase = t / max(1e-6, seconds)
        s = np.array([mq/EQ_SCL, mx/EX_SCL, mt/MT_SCL,
                      phase, math.sin(2*math.pi*phase), math.cos(2*math.pi*phase),
                      *gp.normalized()], dtype=float)  # 3 + 3 + 4 = 10
        a, logp, ent, mu, std = policy.act(s, rng)
        if k % max(1, action_stride) == 0:
            a, logp, ent, mu, std = policy.act(s, rng)
            gp.step(a, frac_step=frac_step)
            last_a, last_mu, last_std = a, mu, std
        else:
            a = last_a; mu = last_mu; std = last_std
            logp = 0.0; ent = 0.0
        # reward (negative cost)
        WQ, WX, WT, WLIM = 1.0, 1.5, 1e-4, 1e-3
        r = -(WQ*mq + WX*mx + WT*mt) - WLIM*np.mean(a*a) + policy.ent_coef*ent
        S_list.append(s); A_list.append(a); R_list.append(r); LOGP.append(logp); MU.append(mu); ENT.append(ent)
        t += tracker.dt_sim
    # compute returns/adv
    S_arr = np.array(S_list); A_arr = np.array(A_list); R_arr = np.array(R_list, dtype=float)
    N = len(R_arr); gamma = 0.995
    ret = np.zeros(N, dtype=float); running = 0.0
    for i in range(N-1, -1, -1):
        running = R_arr[i] + gamma*running; ret[i] = running
    V = np.array([policy.value(s) for s in S_arr], dtype=float)
    adv = ret - V
    batch = {"s": S_arr, "a": A_arr, "ret": ret, "adv": adv,
             "mu": np.array(MU), "std": np.exp(policy.log_std), "logp": np.array(LOGP)}
    info = {"mq": getattr(tracker, "_log_mq", []).copy(),
            "mx": getattr(tracker, "_log_mx", []).copy(),
            "mt": getattr(tracker, "_log_mt", []).copy(),
            "t":  getattr(tracker, "_log_t",  []).copy()}
    return batch, info

def train_loop(args):
    # load config/assets
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    qpos_ref, xpos_ref = prep_refs_from_pkl(Path(args.traj))
    tracker = TrajectoryTracker(model, data, qpos_ref, config, xpos_ref=None if not args.use_xref else xpos_ref)

    # policy + params
    state_dim = 10; action_dim = 4
    policy = LinearGaussianPolicy(state_dim, action_dim, init_std=0.08, seed=args.seed, ent_coef=1e-3, lr=1e-2)
    rng = np.random.default_rng(args.seed)
    # bounds for multipliers

    best_J = -1e18
    for ep in range(1, args.train_episodes+1):
        tracker.reset_to_start()
        gp = GainParam4(lo=[0.2, 0.2, 0.2, 0.2], hi=[5.0, 5.0, 5.0, 5.0])
        batch, info = rollout_episode(tracker, args.seconds, gp, policy, rng, args.action_stride, args.frac_step)
        policy.update(batch)
        J = - (1.0*np.mean(info["mq"]) + 0.5*np.mean(info["mx"]) + 1e-4*np.mean(info["mt"]))
        if J > best_J:
            best_J = J
            save_policy(policy, Path("rl_gains_policy.npz"))
        print(f"[TRAIN] ep={ep:03d} J={J:.4e} mq={np.mean(info['mq']):.3e} mx={np.mean(info['mx']):.3e} mt={np.mean(info['mt']):.3e}")

# def eval_loop(args):
#     with open(args.cfg, "r") as f:
#         config = yaml.safe_load(f)
#     model = mujoco.MjModel.from_xml_path(args.xml)
#     data  = mujoco.MjData(model)
#     qpos_ref, xpos_ref = prep_refs_from_pkl(Path(args.traj))
#     tracker = TrajectoryTracker(model, data, qpos_ref, config, xpos_ref=None if not args.use_xref else xpos_ref)
#     pol = load_policy(Path("rl_gains_policy.npz"))
#     rng = np.random.default_rng(args.seed+123)
#     # reset gains holder
#     gp = GainParam4(lo=[0.2, 0.2, 0.2, 0.2], hi=[5.0, 5.0, 5.0, 5.0])
#     tracker.reset_to_start()
#     batch, info = rollout_episode(tracker, args.seconds, gp, pol, rng)
#     # write metrics
#     with open("rl_eval_metrics.csv","w",newline="",encoding="utf-8") as f:
#         w = csv.writer(f); w.writerow(["t","mq","mx","mt"])
#         for i in range(len(info["t"])):
#             w.writerow([info["t"][i], info["mq"][i], info["mx"][i], info["mt"][i]])
#     with open("rl_eval_summary.txt","w",encoding="utf-8") as f:
#         f.write(f"mq_avg={np.mean(info['mq']):.6e}\n")
#         f.write(f"mx_avg={np.mean(info['mx']):.6e}\n")
#         f.write(f"mt_avg={np.mean(info['mt']):.6e}\n")
#     print("[EVAL] wrote rl_eval_metrics.csv, rl_eval_summary.txt")

def eval_loop(args):
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    qpos_ref, xpos_ref = prep_refs_from_pkl(Path(args.traj))
    tracker = TrajectoryTracker(model, data, qpos_ref, config)
    pol = load_policy(Path("rl_gains_policy.npz"))
    rng = np.random.default_rng(args.seed+123)
    gp = GainParam4(lo=[0.2, 0.2, 0.2, 0.2], hi=[5.0, 5.0, 5.0, 5.0])
    tracker.reset_to_start()
    policy_state = {
        "k": 0,
        "last_a": np.zeros(4, dtype=float),
        "last_mu": np.zeros(4, dtype=float),
        "last_std": np.exp(pol.log_std)
    }

    if not args.show_viewer:
        batch, info = rollout_episode(tracker, args.seconds, gp, pol, rng)
        with open("rl_eval_metrics.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["t","mq","mx","mt"])
            for i in range(len(info["t"])):
                w.writerow([info["t"][i], info["mq"][i], info["mx"][i], info["mt"][i]])
        with open("rl_eval_summary.txt","w",encoding="utf-8") as f:
            f.write(f"mq_avg={np.mean(info['mq']):.6e}\n")
            f.write(f"mx_avg={np.mean(info['mx']):.6e}\n")
            f.write(f"mt_avg={np.mean(info['mt']):.6e}\n")
        print("[EVAL] wrote rl_eval_metrics.csv, rl_eval_summary.txt")
        return

    def _step_with_policy(tracker, t, gp, pol, rng):
        alpha_p, alpha_d, beta_x, beta_v = gp.values()
        tracker.alpha_p = alpha_p; tracker.alpha_d = alpha_d
        tracker.beta_x  = beta_x;  tracker.beta_v  = beta_v
        mq, mx, mt = tracker.step_once(t)
        EQ_SCL, EX_SCL, MT_SCL = 0.2, 0.03, 500.0
        phase = t / max(1e-6, args.seconds)
        s = np.array([mq/EQ_SCL, mx/EX_SCL, mt/MT_SCL,
                      phase, math.sin(2*math.pi*phase), math.cos(2*math.pi*phase),
                      *gp.normalized()], dtype=float)
        k = policy_state["k"]
        if k % max(1, args.action_stride) == 0:
            a, _, _, mu, std = pol.act(s, rng)
            gp.step(a, frac_step=args.frac_step)
            policy_state["last_a"]  = a
            policy_state["last_mu"] = mu
            policy_state["last_std"]= std
        else:
            a = policy_state["last_a"]

        policy_state["k"] = k + 1
        return mq, mx, mt

    # 实时可视化
    if args.show_viewer:
        import mujoco.viewer as viewer
        t = 0.0
        with viewer.launch_passive(model, data) as v:
            while t < args.seconds and t < tracker.t_total:
                if not v.is_running():
                    break
                _ = _step_with_policy(tracker, t, gp, pol, rng)
                v.sync()
                time.sleep(1/50)
                t += tracker.dt_sim
        print("[EVAL] viewer done.")
        return

def save_policy(policy, path: Path):
    np.savez(path, W=policy.W, b=policy.b, log_std=policy.log_std, v=policy.v, c=policy.c)

def load_policy(path: Path):
    z = np.load(path, allow_pickle=False)
    state_dim = z["W"].shape[1]; action_dim = z["W"].shape[0]
    pol = LinearGaussianPolicy(state_dim, action_dim)
    pol.W = z["W"]; pol.b = z["b"]; pol.log_std = z["log_std"]
    pol.v = z["v"]; pol.c = float(z["c"])
    return pol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--traj", type=str, required=True)
    ap.add_argument("--cfg", type=str, default="cfg.yaml")
    ap.add_argument("--use_xref", action="store_true")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--train_episodes", type=int, default=200)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_viewer", action="store_true")
    ap.add_argument("--action_stride", type=int, default=5)
    ap.add_argument("--frac_step", type=float, default=0.05)
    args = ap.parse_args()

    if args.eval:
        eval_loop(args)
    else:
        train_loop(args)

if __name__ == "__main__":
    main()
