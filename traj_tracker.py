from typing import Optional, List
import pickle
import numpy as np
import mujoco
import math
import mujoco.viewer as viewer

from utils import (
    load_config, as_vec, to_mat3,
    rotmat_to_rotvec,
    sample_ref, sample_rot, compute_qvel_qacc
)
from metrics import MetricsLogger


class TrajectoryTracker:
    REF_FPS = 50.0
    REF_DT  = 1.0 / REF_FPS

    def __init__(self, model, data, qpos_ref, config: dict, xpos_ref: Optional[np.ndarray] = None):
        self.model = model
        self.data  = data
        self.nq = model.nq; self.nv = model.nv
        assert self.nq == self.nv
        self._scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # refs
        self.qpos_ref = np.asarray(qpos_ref, float)             # (T, nq)
        self.T = self.qpos_ref.shape[0]
        self.t_total = (self.T - 1) * self.REF_DT
        self.qvel_ref, self.qacc_ref = compute_qvel_qacc(model, self.qpos_ref, self.REF_DT)

        # config
        get = lambda k, d: config[k] if k in config else d
        self.task_enabled         = bool(get("task_enabled", True))
        self.task_mode            = get("task_mode", "site")         # 'body' or 'site'
        self.task_names: List[str]= list(get("task_names", []))
        self.use_orientation_task = bool(get("use_orientation_task", False))
        self.torque_limit         = as_vec(get("torque_limit", 1500.0), self.nv)

        # gains
        self.kp_q = as_vec(get("kp_q", 60.0), self.nv)
        self.kd_q = as_vec(get("kd_q", 12.0), self.nv)
        self.kx   = get("kx", 500.0)
        self.dx   = get("dx", 50.0)
        self.kr   = get("kr", 200.0)
        self.dr   = get("dr", 20.0)
        
        self.alpha_p = 1.0  # scales kp_q
        self.alpha_d = 1.0  # scales kd_q
        self.beta_x  = 1.0  # scales kx
        self.beta_v  = 1.0  # scales dx

        # task references (pos/vel [+ rot/angvel])
        self.task_ids = self._resolve_task_ids(self.task_mode, self.task_names)
        self.xpos_ref, self.xvel_ref, self.orient_ref, self.angvel_ref = self._build_task_refs(
            self.task_mode, self.task_ids, xpos_ref
        )

        # sim dt
        self.dt_sim = model.opt.timestep
        
        self._log_t:  list[float] = []
        self._log_mq: list[float] = []
        self._log_mx: list[float] = []
        self._log_mt: list[float] = []
        self._last_tau = np.zeros(self.nv, dtype=float)

    # ---------- task id resolution ----------
    def _resolve_task_ids(self, mode: str, names: List[str]) -> List[int]:
        if not self.task_enabled: return []
        ids: List[int] = []
        if mode == 'body':
            all_names = [self.model.body(i).name for i in range(self.model.nbody)]
            for nm in names:
                assert nm in all_names, f"unknown body: {nm}"
                bid = all_names.index(nm); assert bid != 0, "do not select worldbody"
                ids.append(bid)
        elif mode == 'site':
            all_names = [self.model.site(i).name for i in range(self.model.nsite)]
            for nm in names:
                assert nm in all_names, f"unknown site: {nm}"
                ids.append(all_names.index(nm))
        else:
            raise ValueError("task_mode must be 'body' or 'site'")
        return ids

    # ---------- build task refs over full traj ----------
    def _fk_stack_over_traj(self, ids: List[int], *, is_body: bool, need_orient: bool):
        d = self._scratch
        S = len(ids)
        X    = np.zeros((self.T, S, 3), dtype=float)
        Xdot = np.zeros_like(X)
        R    = np.zeros((self.T, S, 3, 3), dtype=float)  if need_orient else None
        W    = np.zeros((self.T, S, 3), dtype=float)     if need_orient else None
        jacp = np.zeros((3, self.nv)); jacr = np.zeros((3, self.nv))

        for t in range(self.T):
            d.qpos[:] = self.qpos_ref[t]; d.qvel[:] = self.qvel_ref[t]
            mujoco.mj_forward(self.model, d)
            for s, tid in enumerate(ids):
                if is_body:
                    X[t, s, :] = d.xpos[tid]
                    mujoco.mj_jacBody(self.model, d, jacp, jacr, tid)
                    if need_orient: R[t, s] = d.xmat[tid].reshape(3, 3)
                else:
                    X[t, s, :] = d.site_xpos[tid]
                    mujoco.mj_jacSite(self.model, d, jacp, jacr, tid)
                    if need_orient: R[t, s] = d.site_xmat[tid].reshape(3, 3)
                Xdot[t, s] = jacp @ d.qvel
                if need_orient: W[t, s] = jacr @ d.qvel
        return X, Xdot, R, W

    def _build_task_refs(self, mode: str, ids: List[int], xpos_ref: Optional[np.ndarray]):
        if (not self.task_enabled) or len(ids) == 0:
            return None, None, None, None
        if xpos_ref is not None:
            assert mode == 'body', "xpos_ref requires task_mode='body'"
            Xall = np.asarray(xpos_ref, float)  # (T, nbody, 3)
            S = len(ids)
            Xsel = np.zeros((self.T, S, 3), dtype=float)
            for s, bid in enumerate(ids):
                assert bid != 0, "task_names/ids cannot include worldbody"
                Xsel[:, s, :] = Xall[:, bid, :]
            Xdot = np.zeros_like(Xsel); R = None; W = None
            return Xsel, Xdot, R, W
        need_orient = bool(self.use_orientation_task)
        return self._fk_stack_over_traj(ids, is_body=(mode == 'body'), need_orient=need_orient)

    # ---------- control terms ----------
    def _tau_inverse(self, t: float) -> np.ndarray:
        q_d   = sample_ref(self.qpos_ref, t, self.REF_DT, self.t_total)
        qd_d  = sample_ref(self.qvel_ref, t, self.REF_DT, self.t_total)
        qdd_d = sample_ref(self.qacc_ref, t, self.REF_DT, self.t_total)
        q  = self.data.qpos.copy(); qd = self.data.qvel.copy()
        e  = q_d - q; de = qd_d - qd
        
        kp_eff = self.alpha_p * self.kp_q
        kd_eff = self.alpha_d * self.kd_q

        v  = qdd_d + kd_eff * de + kp_eff * e
        d = self._scratch; d.qpos[:] = q; d.qvel[:] = qd; d.qacc[:] = v
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def _tau_task(self, t: float) -> np.ndarray:
        if (not self.task_enabled) or (len(self.task_ids) == 0):
            return np.zeros(self.nv)
        x_d = sample_ref(self.xpos_ref,   t, self.REF_DT, self.t_total)
        v_d = sample_ref(self.xvel_ref,   t, self.REF_DT, self.t_total)
        R_d = sample_rot(self.orient_ref, t, self.REF_DT, self.t_total) if self.use_orientation_task else None
        w_d = sample_ref(self.angvel_ref, t, self.REF_DT, self.t_total) if self.use_orientation_task else None

        tau_task = np.zeros(self.nv)
        jacp = np.zeros((3, self.nv)); jacr = np.zeros((3, self.nv))
        mujoco.mj_fwdPosition(self.model, self.data)
        qd = self.data.qvel
        Kx = to_mat3(self.beta_x * self.kx)
        Dx = to_mat3(self.beta_v * self.dx)
        KR = to_mat3(self.kr)
        DR = to_mat3(self.dr)

        for s, tid in enumerate(self.task_ids):
            if self.task_mode == 'body':
                x = self.data.xpos[tid].copy()
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, tid)
                R_now = self.data.xmat[tid].reshape(3, 3)
            else:
                x = self.data.site_xpos[tid].copy()
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, tid)
                R_now = self.data.site_xmat[tid].reshape(3, 3)

            v = jacp @ qd
            ex = (x_d[s] - x) if (x_d is not None) else (-x)
            ev = (v_d[s] - v) if (v_d is not None) else (-v)
            tau_task += jacp.T @ (Kx @ ex + Dx @ ev)

            if self.use_orientation_task and (R_d is not None):
                eR = rotmat_to_rotvec(R_d[s] @ R_now.T)
                w  = jacr @ qd
                ew = (w_d[s] - w) if (w_d is not None) else (-w)
                tau_task += jacr.T @ (KR @ eR + DR @ ew)

        return tau_task

    def compute_tau(self, t: float) -> np.ndarray:
        tau = self._tau_inverse(t) + self._tau_task(t)
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    # ---------- one sim step (with optional metrics hook) ----------
    def step_once(self, t: float, metrics: Optional[MetricsLogger] = None):
        tau = self.compute_tau(t)
        
        q_d = sample_ref(self.qpos_ref, t, self.REF_DT, self.t_total)
        e   = q_d - self.data.qpos
        mq  = float(np.linalg.norm(e) / math.sqrt(max(1, e.size)))
        mt  = float(np.mean(np.abs(tau)))
        mx = 0.0
        if (self.xpos_ref is not None) and (len(self.task_ids) > 0):
            x_d = sample_ref(self.xpos_ref, t, self.REF_DT, self.t_total)
            if x_d is not None:
                X_now = []
                for tid in self.task_ids:
                    if self.task_mode == 'site':
                        X_now.append(self.data.site_xpos[tid])
                    else:
                        X_now.append(self.data.xpos[tid])
                X_now = np.asarray(X_now)
                err_norm = np.linalg.norm(x_d - X_now, axis=1)
                mx = float(err_norm.mean())
        
        self._last_tau = tau.copy()
        self._log_t.append(float(t)); self._log_mq.append(mq); self._log_mx.append(mx); self._log_mt.append(mt)
        if metrics is not None:
            metrics.update(self, t, tau, mq=mq, mx=mx, mt=mt)
    
        self.data.qfrc_applied[:] = tau
        mujoco.mj_step(self.model, self.data)

        return mq, mx, mt
    
    def reset_to_start(self):
        self.data.qpos[:] = self.qpos_ref[0]
        self.data.qvel[:] = self.qvel_ref[0]
        mujoco.mj_forward(self.model, self.data)
        self._log_t.clear(); self._log_mq.clear(); self._log_mx.clear(); self._log_mt.clear()

    def reset_to_time(self, t0: float):
        q0  = sample_ref(self.qpos_ref, t0, self.REF_DT, self.t_total)
        qd0 = sample_ref(self.qvel_ref, t0, self.REF_DT, self.t_total)
        self.data.qpos[:] = q0
        self.data.qvel[:] = qd0
        mujoco.mj_forward(self.model, self.data)
        self._log_t.clear(); self._log_mq.clear(); self._log_mx.clear(); self._log_mt.clear()

    # ---------- top-level run (config-driven, no argparse) ----------
    def run_from_cfg(self, cfg: dict):
        # assets/run options
        run_cfg = cfg.get("run", {}) if isinstance(cfg, dict) else {}
        show_viewer = bool(run_cfg.get("show_viewer", True))
        seconds = run_cfg.get("seconds", None)

        # optional truncation by seconds
        if isinstance(seconds, (int, float)):
            T_keep = int(min(self.T - 1, round(seconds * self.REF_FPS))) + 1
            self.qpos_ref = self.qpos_ref[:T_keep]
            self.qvel_ref = self.qvel_ref[:T_keep]
            self.qacc_ref = self.qacc_ref[:T_keep]
            self.T = self.qpos_ref.shape[0]
            self.t_total = (self.T - 1) * self.REF_DT

        # metrics
        metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
        metrics = MetricsLogger(metrics_cfg)

        # reset state to start of ref
        self.reset_to_start()

        # loop
        t = 0.0
        if show_viewer:
            import time
            with viewer.launch_passive(self.model, self.data) as v:
                while t <= self.t_total:
                    if not v.is_running(): break
                    self.step_once(t, metrics=metrics)
                    v.sync()
                    time.sleep(1/100)
                    t += self.dt_sim
        else:
            steps = int(np.ceil(self.t_total / self.dt_sim))
            for _ in range(steps):
                self.step_once(t, metrics=metrics)
                t += self.dt_sim

        metrics.summarize_and_maybe_write_csv()
        return metrics  # for programmatic access (optional)


# -------- convenience entry (only cfg.yaml needed) --------
def main_from_cfg(cfg_path="cfg.yaml"):
    cfg = load_config(cfg_path)
    xml_path  = cfg.get("assets", {}).get("xml",  "RKOB_simplified_upper_with_marker.xml")
    traj_path = cfg.get("assets", {}).get("traj", "AJ026_Run_Comfortable.pkl")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    qpos_ref = np.asarray(traj["mj_qpos"], float)
    xpos_ref = traj.get("mj_xpos", None)

    tracker = TrajectoryTracker(model, data, qpos_ref, cfg)
    tracker.run_from_cfg(cfg)


if __name__ == "__main__":
    main_from_cfg()
