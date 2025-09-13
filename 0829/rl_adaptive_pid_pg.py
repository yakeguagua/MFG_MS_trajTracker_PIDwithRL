#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_adaptive_pid_pg.py
在线自适应 PID/任务空间增益（Kp_tau, Kd_tau, KX, DX），RL 在每个控制步输出对增益的微调量（log 空间），
并限制在指定边界内。无需第三方 RL 库：线性高斯策略 + REINFORCE（回报基线为线性值函数回归）。

用法：
# 训练（默认 200 回合，可按需增减）
python -u rl_adaptive_pid_pg.py --xml RKOB_simplified_upper_with_marker.xml \
  --qpos qpos_hist.csv --xpos xpos_hist.csv --use_xref --seconds 6.0 --train_episodes 200

# 仅评估（加载已训练策略，在线自适应）
python -u rl_adaptive_pid_pg.py --xml RKOB_simplified_upper_with_marker_floor_down.xml \
  --qpos qpos_hist.csv --xpos xpos_hist.csv --use_xref --seconds 6.0 --eval

产物：
- rl_gains_policy.npz     （策略、值函数、边界等）
- rl_eval_metrics.csv     （eval 时的逐帧 mq/mx/mt）
- rl_eval_summary.txt     （eval 平均指标）
"""

import argparse, json, math, csv
from pathlib import Path
import numpy as np
import mujoco

# ========================= 基本配置 =========================
FPS_TRAJ   = 60.0
DT_SIM     = 5e-4           # mujoco 模拟步长
TORQUE_LIM = 1500.0
RAMP_T     = 0.3            # 前几百毫秒淡入（可在奖励里不做特殊处理）

# 增益边界（与你全局/局部搜索保持一致）
BOUNDS = {
    "Kp_tau": (20.0, 200.0),
    "Kd_tau": (5.0, 40.0),
    "KX":     (200.0, 2000.0),
    "DX":     (10.0, 120.0),
}

# 奖励权重（负代价）
WQ   = 1.0       # 关节 RMS 误差权重
WX   = 0.5       # 任务空间误差权重
WT   = 1e-4      # 力矩幅值权重
WLIM = 1e-3      # 每步增益变化惩罚（过大导致频繁抖动）
WENT = 1e-3      # 策略熵系数（鼓励探索）

# ========================= 轨迹/模型工具 =========================
def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 CSV: {p.resolve()}")
    arr = np.loadtxt(p, delimiter=",")
    if arr.ndim != 2: raise ValueError("CSV 应为二维 [T, N]")
    return arr

def prep_refs(qpos_csv, xpos_csv=None):
    qpos_ref = load_csv(qpos_csv)
    T, nq = qpos_ref.shape
    dt_ref = 1.0 / FPS_TRAJ
    qd_ref  = np.gradient(qpos_ref, dt_ref, axis=0)
    qdd_ref = np.gradient(qd_ref,   dt_ref, axis=0)

    xpos_ref = None
    HAS_XREF = False
    if xpos_csv and Path(xpos_csv).exists():
        xr = load_csv(xpos_csv)
        if xr.shape[0] == T and xr.shape[1] % 3 == 0:
            nb = xr.shape[1] // 3
            xpos_ref = xr.reshape(T, nb, 3)
            HAS_XREF = True
    return qpos_ref, qd_ref, qdd_ref, dt_ref, HAS_XREF, xpos_ref

def time_to_index(sim_t, dt_ref, T):
    return int(np.clip(np.floor(sim_t / dt_ref), 0, T-1))

class ControllerSim:
    SITE_CANDS = ("toes_r","r_toe","foot_r","r_foot","right_toe","right_foot")
    BODY_CANDS = ("calcn_r","toes_r","r_foot","foot_r","right_foot")
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.data_fk = mujoco.MjData(self.model)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        self.model.opt.timestep   = DT_SIM
        # 目标（末端/足底）
        self.TARGET_IS_SITE = False
        self.TARGET_ID = -1
        for nm in self.SITE_CANDS:
            try:
                self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, nm)
                self.TARGET_IS_SITE = True; break
            except Exception: pass
        if self.TARGET_ID < 0:
            for nm in self.BODY_CANDS:
                try:
                    self.TARGET_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                    self.TARGET_IS_SITE = False; break
                except Exception: pass

    def reset_state(self, q0, qd0):
        self.data.qpos[:] = q0
        self.data.qvel[:] = qd0
        mujoco.mj_forward(self.model, self.data)

    def step_once(self, sim_t, qpos_ref, qd_ref, qdd_ref, dt_ref, gains, use_xref=False, xpos_ref=None):
        """
        返回：eq_rms, ex_norm, mean|tau|
        """
        Kp_tau, Kd_tau, KX, DX = gains
        T = qpos_ref.shape[0]
        i = time_to_index(sim_t, dt_ref, T)
        q_d, qd_d, qdd_d = qpos_ref[i], qd_ref[i], qdd_ref[i]
        q, qd = self.data.qpos.copy(), self.data.qvel.copy()
        e, de = q_d - q, qd_d - qd

        # 完整逆动力学前馈
        self.data.qacc[:] = qdd_d
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse.copy()

        # 关节域 PD
        tau += Kp_tau * e + Kd_tau * de

        # 任务空间 PD（如果找到了目标）
        ex_norm = 0.0
        if self.TARGET_ID >= 0:
            Jpos = np.zeros((3, self.model.nv))
            if self.TARGET_IS_SITE:
                x = self.data.site_xpos[self.TARGET_ID]
                mujoco.mj_jacSite(self.model, self.data, Jpos, None, self.TARGET_ID)
            else:
                x = self.data.xpos[self.TARGET_ID]
                mujoco.mj_jacBody(self.model, self.data, Jpos, None, self.TARGET_ID)
            xdot = Jpos @ self.data.qvel
            # 期望 x 通过 q_d 的 FK
            self.data_fk.qpos[:] = q_d
            self.data_fk.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data_fk)
            x_d = self.data_fk.site_xpos[self.TARGET_ID] if self.TARGET_IS_SITE else self.data_fk.xpos[self.TARGET_ID]
            xdot_d = np.zeros(3)
            f_task = KX * (x_d - x) + DX * (xdot_d - xdot)
            tau += Jpos.T @ f_task
            ex_norm = float(np.linalg.norm(x_d - x))

        tau = np.clip(tau, -TORQUE_LIM, TORQUE_LIM)
        self.data.qfrc_applied[:] = tau
        mujoco.mj_step(self.model, self.data)

        eq_rms = float(np.linalg.norm(e) / max(1.0, math.sqrt(e.size)))
        mt     = float(np.mean(np.abs(tau)))
        return eq_rms, ex_norm, mt

# ========================= 增益 param（log 空间） =========================
class GainParam:
    """
    持有 log-增益向量 y，真实增益 g = clip(exp(y), lo, hi)
    action 是 Δy（每步限幅），这样天然正值 + 边界稳定。
    """
    def __init__(self, bounds_dict):
        self.keys = ["Kp_tau","Kd_tau","KX","DX"]
        self.lo   = np.array([bounds_dict[k][0] for k in self.keys], dtype=float)
        self.hi   = np.array([bounds_dict[k][1] for k in self.keys], dtype=float)
        self.y_lo = np.log(self.lo)
        self.y_hi = np.log(self.hi)
        # 初始放到对数区间中点
        self.y    = 0.5*(self.y_lo + self.y_hi)

    def set_from_gains(self, gains):
        g = np.array(gains, dtype=float)
        self.y = np.log(np.clip(g, self.lo, self.hi))

    def step(self, dy, frac_step=0.05):
        # 每步 Δy 限幅到区间宽度的一定比例
        max_step = frac_step * (self.y_hi - self.y_lo)
        dy = np.clip(dy, -max_step, max_step)
        self.y = np.clip(self.y + dy, self.y_lo, self.y_hi)

    def gains(self):
        return np.clip(np.exp(self.y), self.lo, self.hi)

    def normalized(self):
        # 归一化到 [0,1]，用于观测
        return (self.y - self.y_lo) / (self.y_hi - self.y_lo + 1e-12)

# ========================= 策略（线性高斯） + 值函数（线性） =========================
class LinearGaussianPolicy:
    def __init__(self, state_dim, action_dim, init_std=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.05, size=(action_dim, state_dim))  # 均值 μ=Ws+b
        self.b = np.zeros(action_dim)
        self.log_std = np.log(init_std) * np.ones(action_dim)          # 固定方差（可训练也行，这里先固定）
        # 线性值函数 V(s)=v^T s + c
        self.v = np.zeros(state_dim)
        self.c = 0.0
        # 学习率
        self.lr_pi = 1e-2
        self.lr_v  = 1e-2
        self.ent_coef = WENT

    def act(self, s, rng):
        mu = self.W @ s + self.b
        std = np.exp(self.log_std)
        a = mu + std * rng.normal(size=mu.shape)
        # log prob（高斯对角）
        logp = -0.5*np.sum(((a-mu)/std)**2 + 2*np.log(std) + np.log(2*np.pi))
        ent  = 0.5*np.sum(np.log(2*np.pi*np.e) + 2*np.log(std))
        return a, float(logp), float(ent), mu, std

    def value(self, s):
        return float(self.v @ s + self.c)

    def update(self, batch):
        """
        batch: dict with np arrays
          s: [N, S], a: [N, A], ret: [N], adv: [N], mu: [N, A], std: [A], logp: [N]
        REINFORCE + baseline，优势标准化后更新；值函数用最小二乘回归一步近似。
        """
        S = batch["s"]; A = batch["a"]; ADV = batch["adv"]
        MU = batch["mu"]; STD = batch["std"]; LOGP = batch["logp"]
        N, state_dim = S.shape
        A_dim = A.shape[1]
        # 标准化优势
        adv = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # π 的梯度：∇θ logN(a|μ,σ) = (a-μ)/σ^2 * ∂μ/∂θ
        # μ = W s + b
        inv_var = 1.0 / (STD**2 + 1e-12)   # [A]
        # 累积梯度
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        # 熵奖励梯度（对 log_std 的梯度，这里固定不更新方差，所以忽略）
        # 逐样本累积
        for n in range(N):
            s = S[n]
            diff = (A[n] - MU[n]) * inv_var  # [A]
            gW += np.outer(diff * adv[n], s)
            gb += diff * adv[n]
        # 熵 regularizer 对 W,b 无梯度，只有对 std 有（此处不更新 std，直接加到目标里即可）
        self.W += self.lr_pi * gW / N
        self.b += self.lr_pi * gb / N

        # 值函数回归：最小化 (V(s)-ret)^2
        # 闭式解（带偏置）：用增广矩阵做一轮最小二乘
        X = np.hstack([S, np.ones((N,1))])         # [N, S+1]
        y = batch["ret"].reshape(-1,1)             # [N,1]
        # 正则稳定
        reg = 1e-4*np.eye(X.shape[1])
        theta = np.linalg.pinv(X.T@X + reg) @ (X.T @ y)   # [S+1,1]
        self.v = theta[:-1,0]
        self.c = theta[-1,0]

# ========================= 训练/评估循环 =========================
def rollout_episode(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, seconds, use_xref, xpos_ref,
                    policy: LinearGaussianPolicy, rng, init_gains=None):
    """
    返回轨迹数据：states, actions, rewards, logp, mu, std, infos(记录mq/mx/mt等)
    """
    T = qpos_ref.shape[0]
    sim.reset_state(qpos_ref[0], qd_ref[0])

    gp = GainParam(BOUNDS)
    if init_gains is not None:
        gp.set_from_gains(init_gains)

    # 观测标准化的尺度（经验值防爆）
    EQ_SCL = 0.2     # rad
    EX_SCL = 0.05    # m
    MT_SCL = 500.0   # Nm

    states=[]; actions=[]; rewards=[]; logps=[]; mus=[]; ents=[]
    infos = {"mq":[],"mx":[],"mt":[],"gKp":[],"gKd":[],"gKX":[],"gDX":[],"t":[]}

    t = 0.0
    # 用 60Hz 的“帧对齐”统计指标，但动作频率用每个模拟步都可以，这里简化：每个模拟步都可动作
    while t < seconds and t < (T-1)*dt_ref:
        # 当前真实增益
        g = gp.gains()
        # 走一步
        mq, mx, mt = sim.step_once(t, qpos_ref, qd_ref, qdd_ref, dt_ref, g, use_xref, xpos_ref)

        # 构造状态向量（紧凑：误差 + 正则化增益 + 归一化时间相位 + 上一次动作幅度可以不加）
        phase = t / max(1e-6, seconds)
        s = np.array([
            mq/EQ_SCL, mx/EX_SCL, mt/MT_SCL,
            phase, math.sin(2*math.pi*phase), math.cos(2*math.pi*phase),
            *gp.normalized()  # 4 维
        ], dtype=float)   # 3 + 3 + 4 = 10 维
        # 动作：Δy（log-gains 的增量）
        a, logp, ent, mu, std = policy.act(s, rng=rng)
        # 应用动作
        gp.step(a, frac_step=0.05)

        # 奖励（负代价 & 平滑增益变化）
        r = -(WQ*mq + WX*mx + WT*mt) - WLIM*np.mean(a*a)
        # 加一点熵作为奖励（鼓励探索）
        r += policy.ent_coef * ent

        # 记录
        states.append(s); actions.append(a); rewards.append(r); logps.append(logp); mus.append(mu); ents.append(ent)
        infos["mq"].append(mq); infos["mx"].append(mx); infos["mt"].append(mt)
        infos["gKp"].append(g[0]); infos["gKd"].append(g[1]); infos["gKX"].append(g[2]); infos["gDX"].append(g[3])
        infos["t"].append(t)

        t += sim.model.opt.timestep

    # 计算 return/advantage（回报到终点；基线为 V(s)）
    S = np.array(states); A = np.array(actions); R = np.array(rewards, dtype=float)
    N = len(R)
    # 折扣因子（越接近 1 把后半段也算进去）
    gamma = 0.995
    ret = np.zeros(N, dtype=float)
    running = 0.0
    for i in range(N-1, -1, -1):
        running = R[i] + gamma*running
        ret[i] = running
    V = np.array([policy.value(s) for s in S], dtype=float)
    adv = ret - V

    batch = {
        "s": S,
        "a": A,
        "ret": ret,
        "adv": adv,
        "mu": np.array(mus),
        "std": np.exp(policy.log_std),
        "logp": np.array(logps)
    }
    return batch, infos

def train(args):
    qpos_ref, qd_ref, qdd_ref, dt_ref, HAS_XREF, xpos_ref = prep_refs(args.qpos, args.xpos if args.use_xref else None)
    use_xref = args.use_xref and HAS_XREF and (xpos_ref is not None)
    sim = ControllerSim(args.xml)
    assert qpos_ref.shape[1] == sim.model.nq, f"CSV 列数({qpos_ref.shape[1]}) 与模型 nq({sim.model.nq}) 不一致"

    state_dim = 10
    action_dim = 4
    policy = LinearGaussianPolicy(state_dim, action_dim, init_std=0.08, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # 初始增益（可用 baseline 或 global 的中点，默认对数区间中点）
    init_gains = None

    best_meanJ = -1e18
    for ep in range(1, args.train_episodes+1):
        batch, info = rollout_episode(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, args.seconds, use_xref, xpos_ref,
                                      policy, rng, init_gains=init_gains)
        # 训练一步
        policy.update(batch)

        # 打印统计
        J = - (WQ*np.mean(info["mq"]) + WX*np.mean(info["mx"]) + WT*np.mean(info["mt"]))
        # J 是“负代价”（越大越好）
        if J > best_meanJ:
            best_meanJ = J
            # 保存策略
            save_policy(policy, Path("rl_gains_policy.npz"))
        print(f"[TRAIN] ep={ep:04d} J={J:.4e}  mq={np.mean(info['mq']):.3e}  mx={np.mean(info['mx']):.3e}  mt={np.mean(info['mt']):.3e}")

    print("训练完成，策略保存在 rl_gains_policy.npz")

def eval_policy(args):
    qpos_ref, qd_ref, qdd_ref, dt_ref, HAS_XREF, xpos_ref = prep_refs(args.qpos, args.xpos if args.use_xref else None)
    use_xref = args.use_xref and HAS_XREF and (xpos_ref is not None)
    sim = ControllerSim(args.xml)
    assert qpos_ref.shape[1] == sim.model.nq, f"CSV 列数({qpos_ref.shape[1]}) 与模型 nq({sim.model.nq}) 不一致"

    pol = load_policy(Path("rl_gains_policy.npz"))
    rng = np.random.default_rng(args.seed+123)

    batch, info = rollout_episode(sim, qpos_ref, qd_ref, qdd_ref, dt_ref, args.seconds, use_xref, xpos_ref,
                                  pol, rng, init_gains=None)

    # 写出逐帧与摘要
    with open("rl_eval_metrics.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t","mq","mx","mt","Kp","Kd","KX","DX"])
        for i in range(len(info["t"])):
            w.writerow([info["t"][i], info["mq"][i], info["mx"][i], info["mt"][i],
                        info["gKp"][i], info["gKd"][i], info["gKX"][i], info["gDX"][i]])
    with open("rl_eval_summary.txt","w",encoding="utf-8") as f:
        f.write(f"mq_avg={np.mean(info['mq']):.6e}\n")
        f.write(f"mx_avg={np.mean(info['mx']):.6e}\n")
        f.write(f"mt_avg={np.mean(info['mt']):.6e}\n")
    print("[EVAL] wrote rl_eval_metrics.csv, rl_eval_summary.txt")

def save_policy(policy: LinearGaussianPolicy, path: Path):
    np.savez(path,
             W=policy.W, b=policy.b, log_std=policy.log_std,
             v=policy.v, c=policy.c,
             bounds_lo=np.array([BOUNDS[k][0] for k in ["Kp_tau","Kd_tau","KX","DX"]], dtype=float),
             bounds_hi=np.array([BOUNDS[k][1] for k in ["Kp_tau","Kd_tau","KX","DX"]], dtype=float))

def load_policy(path: Path) -> LinearGaussianPolicy:
    z = np.load(path, allow_pickle=False)
    # state_dim 与 action_dim 在这里按保存尺寸自动推断
    state_dim = z["W"].shape[1]
    action_dim = z["W"].shape[0]
    pol = LinearGaussianPolicy(state_dim, action_dim)
    pol.W = z["W"]; pol.b = z["b"]; pol.log_std = z["log_std"]
    pol.v = z["v"]; pol.c = float(z["c"])
    return pol

# ========================= CLI =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",   type=str, required=True)
    ap.add_argument("--qpos",  type=str, required=True)
    ap.add_argument("--xpos",  type=str, default="")
    ap.add_argument("--use_xref", action="store_true")
    ap.add_argument("--seconds", type=float, default=6.0)

    ap.add_argument("--train_episodes", type=int, default=200)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.eval:
        eval_policy(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
