# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:58:09 2025

@author: YAKE
"""

import mujoco
import mujoco.viewer as viewer
import pickle, time, yaml
import numpy as np
import matplotlib.pyplot as plt

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    
def _project_to_SO3(R):
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj

def rotmat_to_rotvec(R, eps=1e-8):
    R = _project_to_SO3(np.asarray(R, float))
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-8:
        w_hat = 0.5 * (R - R.T)
        return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

    if np.pi - theta < 1e-6:
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            x = axis[0]; y = A[0,1]/(x+eps); z = A[0,2]/(x+eps)
            v = np.array([x,y,z])
        elif axis[1] >= axis[0] and axis[1] >= axis[2]:
            y = axis[1]; x = A[0,1]/(y+eps); z = A[1,2]/(y+eps)
            v = np.array([x,y,z])
        else:
            z = axis[2]; x = A[0,2]/(z+eps); y = A[1,2]/(z+eps)
            v = np.array([x,y,z])
        v = v / (np.linalg.norm(v) + eps)
        return theta * v
    
    w_hat = 0.5 * (R - R.T)
    w = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])
    return (theta / (np.sin(theta) + eps)) * w

def rotvec_to_rotmat(w):
    w = np.asarray(w, float)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)

    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return _project_to_SO3(R)

def so3_log(R):
    return rotmat_to_rotvec(R)

def so3_exp(w):
    return rotvec_to_rotmat(w)

def slerp_R(R0, R1, alpha):
    R0 = _project_to_SO3(R0)
    R1 = _project_to_SO3(R1)
    Rrel = R0.T @ R1
    w = so3_log(Rrel)
    return R0 @ so3_exp(alpha * w)

class TrajectoryTracker:
    REF_FPS = 50.0
    REF_DT  = 1.0 / REF_FPS

    def __init__(self, model, data, qpos_ref, config, xpos_ref=None):
        self.model = model
        self.data = data
        self.nq = model.nq  #广义坐标系数量 qpos
        self.nv = model.nv  #自由度数 qvel
        assert self.nq == self.nv
        self._scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        self.dt_sim = model.opt.timestep
        self.qpos_ref = np.asarray(qpos_ref, float)  # (T, nq)
        self.T = self.qpos_ref.shape[0]
        self.t_total = (self.T - 1) * self.REF_DT

        get = lambda k, d: config[k] if k in config else d
        self.task_enabled         = bool(get("task_enabled", True))
        self.task_mode            = get("task_mode", "body")  # 'body' or 'site'
        self.task_names           = list(get("task_names", []))
        self.use_orientation_task = bool(get("use_orientation_task", False))
        
        # Jnt space PD gain params
        self.kp_q = self._as_vec(get("kp_q", 60.0), self.nv)
        self.kd_q = self._as_vec(get("kd_q", 12.0), self.nv)
        
        self.kx = get("kx", 500.0)
        self.dx = get("dx", 50.0)
        self.kr = get("kr", 200.0)
        self.dr = get("dr", 20.0)
        
        self.torque_limit = self._as_vec(get("torque_limit", 200.0), self.nv)
        
        self.qvel_ref, self.qacc_ref = self._compute_qvel_qacc(self.qpos_ref, self.REF_DT)
        
        self.task_ids = self._resolve_task_ids(self.task_mode, self.task_names)
        self.xpos_ref, self.xvel_ref, self.orient_ref, self.angvel_ref = self._build_task_refs(self.task_mode, self.task_ids, xpos_ref)
        
        self._log_t = []
        self._log_err_mean = []     # 平均绝对误差 mean(|e|)
        self._log_err_max = []      # 最大绝对误差 max(|e|)
        self._log_pelvis_F = []     # tau[:3] 平移力
        self._log_pelvis_M = []     # tau[3:6] 转矩
        self._log_pelvis_wrench_norm = []
        
    @staticmethod
    def _as_vec(x, n):
        if np.isscalar(x): return np.full(n, float(x))
        x = np.asarray(x, float)
        assert x.shape == (n,), f"期望向量形状({n},)"
        return x
    
    @staticmethod
    def _to_mat3(K):
        if np.isscalar(K): return np.eye(3) * float(K)
        K = np.asarray(K, float)
        assert K.shape == (3,3), "K/D/旋转增益需为标量或(3,3)矩阵"
        return K
    
    def _compute_qvel_qacc(self, qpos_ref, dt):
        qpos_proc = qpos_ref.copy()
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

    def _build_task_refs(self, mode, ids, xpos_ref):
        """
        return:
            xpos_ref   : (T, S, 3)
            xvel_ref   : (T, S, 3)
            orient_ref : (T, S, 3, 3) or None
            angvel_ref : (T, S, 3)    or None
        """
        if (not self.task_enabled) or len(ids) == 0:
            return None, None, None, None

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
            R = None
            W = None
            return Xsel, Xdot, R, W
        
        need_orient = bool(self.use_orientation_task)
        X, Xdot, R, W = self._fk_stack_over_traj(ids, is_body=(mode == 'body'), need_orient=need_orient)
        return X, Xdot, R, W
    
    def _fk_stack_over_traj(self, ids, *, is_body, need_orient=False):
        """
        对整条 qpos_ref 生成所选 body/site 的参考：
            x_ref   : (T, S, 3)           世界系位置
            xd_ref: (T, S, 3)            世界系线速度（Jp @ qvel）
            R_ref   : (T, S, 3, 3) or None 世界系方向（仅当 need_orient=True）
            w_ref   : (T, S, 3)    or None 世界系角速度（Jr @ qvel）
        说明：
            - 对 body：位置来自 d.xpos[bid]，方向来自 d.xmat[bid]（reshape 到 3×3，行主序）
            - 对 site：位置来自 d.site_xpos[sid]，方向来自 d.site_xmat[sid]
            - 角速度使用 jacr @ qvel_ref，MuJoCo 的 mj_jacBody/mj_jacSite 返回的 jacr 即角速度雅可比
        """
        d = self._scratch
        S = len(ids)
        X    = np.zeros((self.T, S, 3), dtype=float)
        Xdot = np.zeros_like(X)
        R    = np.zeros((self.T, S, 3, 3), dtype=float)  if need_orient else None
        W    = np.zeros((self.T, S, 3), dtype=float)     if need_orient else None
    
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
    
        for t in range(self.T):
            d.qpos[:] = self.qpos_ref[t]
            d.qvel[:] = self.qvel_ref[t]
            mujoco.mj_forward(self.model, d)
    
            for s, tid in enumerate(ids):
                if is_body:
                    X[t, s, :] = d.xpos[tid]
                    mujoco.mj_jacBody(self.model, d, jacp, jacr, tid)
                    if need_orient:
                        R[t, s, :, :] = d.xmat[tid].reshape(3, 3)
                else:
                    X[t, s, :] = d.site_xpos[tid]
                    mujoco.mj_jacSite(self.model, d, jacp, jacr, tid)
                    if need_orient:
                        R[t, s, :, :] = d.site_xmat[tid].reshape(3, 3)
    
                Xdot[t, s, :] = jacp @ d.qvel
                if need_orient:
                    W[t, s, :] = jacr @ d.qvel
    
        return X, Xdot, R, W
    
    def _sample_ref(self, arr, t):
        if arr is None: return None
        if t <= 0.0: return arr[0]
        if t >= self.t_total: return arr[-1]
        u = t / self.REF_DT
        i0 = int(np.floor(u)); a = u - i0
        return (1-a)*arr[i0] + a*arr[i0+1]
    
    def _sample_rot(self, Rseq, t):
        if Rseq is None:
            return None
        if t <= 0.0:
            return Rseq[0]
        if t >= self.t_total:
            return Rseq[-1]
    
        u = t / self.REF_DT
        i0 = int(np.floor(u))
        a = u - i0  # alpha in [0,1)
    
        R0 = Rseq[i0]     # (S, 3, 3)
        R1 = Rseq[i0 + 1] # (S, 3, 3)
    
        S = R0.shape[0]
        Rout = np.empty_like(R0)
        for s in range(S):
            Rout[s] = slerp_R(R0[s], R1[s], a)
        return Rout
    
    def _log_step(self, t, tau_applied):
        q_ref = self._sample_ref(self.qpos_ref, t)
        e = q_ref - self.data.qpos
        ae = np.abs(e)
    
        self._log_t.append(float(t))
        self._log_err_mean.append(float(ae.mean()))
        self._log_err_max.append(float(ae.max()))
        
        F = tau_applied[:3].copy()
        M = tau_applied[3:6].copy()
        self._log_pelvis_F.append(F)
        self._log_pelvis_M.append(M)
        
        self._log_pelvis_wrench_norm.append(float(np.linalg.norm(F)))
        
    def _plot_tracking_logs(self):
        t = np.asarray(self._log_t, float)
        err_mean = np.asarray(self._log_err_mean, float)
        err_max  = np.asarray(self._log_err_max, float)
        F = np.vstack(self._log_pelvis_F) if len(self._log_pelvis_F) else np.zeros((0,3))
        M = np.vstack(self._log_pelvis_M) if len(self._log_pelvis_M) else np.zeros((0,3))
        Wn = np.asarray(self._log_pelvis_wrench_norm, float)
    
        # Fig 1: 误差
        plt.figure(figsize=(8,4))
        plt.plot(t, err_mean, label="mean(|e|)")
        plt.plot(t, err_max,  label="max(|e|)")
        plt.xlabel("Time [s]"); plt.ylabel("Joint error (abs)")
        plt.title("Joint-space tracking error (abs)")
        plt.legend(); plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.tight_layout()
    
        # # Fig 2: pelvis 平移力 F_x,F_y,F_z
        # if F.size:
        #     plt.figure(figsize=(9,4))
        #     plt.plot(t, F[:,0], label="Fx (pelvis)")
        #     plt.plot(t, F[:,1], label="Fy (pelvis)")
        #     plt.plot(t, F[:,2], label="Fz (pelvis)")
        #     plt.xlabel("Time [s]"); plt.ylabel("Force")
        #     plt.title("Pelvis translational forces (applied)")
        #     plt.legend(ncol=3); plt.grid(True, linewidth=0.5, alpha=0.4)
        #     plt.tight_layout()
    
        # # Fig 3: pelvis 转矩 M_x,M_y,M_z
        # if M.size:
        #     plt.figure(figsize=(9,4))
        #     plt.plot(t, M[:,0], label="Mx (pelvis)")
        #     plt.plot(t, M[:,1], label="My (pelvis)")
        #     plt.plot(t, M[:,2], label="Mz (pelvis)")
        #     plt.xlabel("Time [s]"); plt.ylabel("Torque")
        #     plt.title("Pelvis rotational torques (applied)")
        #     plt.legend(ncol=3); plt.grid(True, linewidth=0.5, alpha=0.4)
        #     plt.tight_layout()
    
        # # Fig 4: 合成范数
        # if Wn.size:
        #     plt.figure(figsize=(8,3.6))
        #     plt.plot(t, Wn, label="||[F]||")
        #     plt.xlabel("Time [s]"); plt.ylabel("Norm")
        #     plt.title("Pelvis wrench norm")
        #     plt.legend(); plt.grid(True, linewidth=0.5, alpha=0.4)
        #     plt.tight_layout()

    # ---------------- 控制分量 ----------------
    def _tau_inverse(self, t):
        q_d   = self._sample_ref(self.qpos_ref, t)
        qd_d  = self._sample_ref(self.qvel_ref, t)
        qdd_d = self._sample_ref(self.qacc_ref, t)
        
        q  = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        
        e  = q_d  - q
        de = qd_d - qd
        v  = qdd_d + self.kd_q * de + self.kp_q * e
        
        d = self._scratch
        d.qpos[:] = q
        d.qvel[:] = qd
        d.qacc[:] = v
        mujoco.mj_inverse(self.model, d)
        
        vmax = float(np.max(np.abs(v)))
        tau_id = d.qfrc_inverse.copy()
        taumax = float(np.max(np.abs(tau_id)))
        #print(f"[tau_inverse t={t:.3f}] |v|_max={vmax:.3f}, |tau_id(mj_inverse)|_max={taumax:.3f}")
    
        return tau_id
    
    def _tau_inverse_with_contact_comp(self, t):
        tau_id = self._tau_inverse(t)
        
        if not hasattr(self, "_scratch_contact"):
            self._scratch_contact = mujoco.MjData(self.model)
            
        dc = self._scratch_contact
        dc.qpos[:] = self.data.qpos
        dc.qvel[:] = self.data.qvel
        dc.qacc[:] = 0.0
        
        mujoco.mj_forward(self.model, dc)
        
        tau_con = dc.qfrc_constraint.copy()
    
        return tau_id + tau_con

    def _tau_task(self, t):
        if (not self.task_enabled) or (len(self.task_ids) == 0):
            return np.zeros(self.nv)
        
        x_d = self._sample_ref(self.xpos_ref,   t)   # (S,3) or None
        v_d = self._sample_ref(self.xvel_ref,   t)   # (S,3) or None
        R_d = self._sample_rot(self.orient_ref, t)   # (S,3,3) or None
        w_d = self._sample_ref(self.angvel_ref, t)   # (S,3) or None

        tau_task = np.zeros(self.nv)
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        
        mujoco.mj_fwdPosition(self.model, self.data)
        
        qd = self.data.qvel
        Kx = self._to_mat3(self.kx)
        Dx = self._to_mat3(self.dx)
        KR = self._to_mat3(self.kr)
        DR = self._to_mat3(self.dr)
        
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
            
            f_lin = Kx @ ex + Dx @ ev
            tau_task += jacp.T @ f_lin
            
            if self.use_orientation_task and (R_d is not None):
                # 旋转误差：eR = log(R_d * R_now^T)
                eR = rotmat_to_rotvec(R_d[s] @ R_now.T)
                # 角速度误差：ew = w_d - w（若 w_d 缺失则仅阻尼）
                w  = jacr @ qd
                ew = (w_d[s] - w) if (w_d is not None) else (-w)
                tau_task += jacr.T @ (KR @ eR + DR @ ew)
    
        return tau_task     

    # ================= 主控循环 =================
    def compute_tau(self, t):
        tau = np.zeros(self.nv)
        tau += self._tau_inverse(t)
        tau += self._tau_task(t)
        tau_clipped = np.clip(tau, -self.torque_limit, self.torque_limit)
    
        if not np.allclose(tau, tau_clipped, rtol=1e-6, atol=1e-8):
            idx = np.where(np.abs(tau) > self.torque_limit + 1e-8)[0]
            print(f"[t={t:.3f}] WARNING: torque clipped at indices {idx.tolist()}")
    
        return tau_clipped

    def step_control(self, t):
        tau = self.compute_tau(t)
        self._log_step(t, tau)
        self.data.qfrc_applied[:] = tau
        mujoco.mj_step(self.model, self.data)

    def run(self, show_viewer=True):
        # 初始化到参考首帧
        self.data.qpos[:] = self.qpos_ref[0]
        self.data.qvel[:] = self.qvel_ref[0]
        mujoco.mj_forward(self.model, self.data)

        t = 0.0
        if show_viewer:
            with viewer.launch_passive(self.model, self.data) as v:
                while t <= self.t_total:
                    if not v.is_running():
                        break
                    self.step_control(t)
                    v.sync()
                    time.sleep(1/500)
                    t += self.dt_sim
                    

        else:
            steps = int(np.ceil(self.t_total / self.dt_sim))
            for _ in range(steps):
                self.step_control(t)
                t += self.dt_sim
        
        self._plot_tracking_logs()
    
with open('cfg.yaml','r') as f:
    config = yaml.safe_load(f)

model = mujoco.MjModel.from_xml_path('RKOB_simplified_upper_with_marker.xml')
data  = mujoco.MjData(model)

with open('AJ026_Run_Comfortable.pkl','rb') as f:
    traj = pickle.load(f)

qpos_ref = np.asarray(traj['mj_qpos'], float)
xpos_ref = traj.get('mj_xpos', None)  

tracker = TrajectoryTracker(model, data, qpos_ref, config)
tracker.run(show_viewer=True)