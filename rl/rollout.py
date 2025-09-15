"""
Episode rollout for online PD-gain tuning with REINFORCE.
- Builds 10D state: [mq/EQ, mx/EX, mt/MT, phase, sin, cos, gains_norm(4)]
- Supports action_stride (hold action for k sim-steps)
- Optional phase randomization (random start time within reference)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from rl.gain_param import GainParam4

@dataclass
class RolloutScales:
    EQ_SCL: float = 0.2
    EX_SCL: float = 0.03
    MT_SCL: float = 500.0
    TAU3_SCL: float = 150.0

@dataclass
class RewardWeights:
    WQ: float = 1.0
    WX: float = 1.0
    WT: float = 1.0e-4
    WT3: float = 0.5
    WLIM: float = 1.0e-3   # action^2 penalty coeff
    WENT: float = 1.0e-3

@dataclass
class RolloutCfg:
    seconds: float
    gamma: float = 0.99
    action_stride: int = 5
    frac_step: float = 0.05
    phase_random: bool = False  # start at a random t0 ∈ [0, t_total-seconds]

def _state_vec(mq, mx, mt, phase, gp: GainParam4, scl: RolloutScales):
    return np.array(
        [mq/scl.EQ_SCL, mx/scl.EX_SCL, mt/scl.MT_SCL,
         phase, math.sin(2*math.pi*phase), math.cos(2*math.pi*phase),
         *gp.normalized()],
        dtype=float
    )  # 3 + 3 + 4 = 10

def rollout_episode(tracker, gp, policy, rng, cfg, scales, rews):
    """
    Run one episode on the given tracker using online-tuned gains.
    Returns:
      batch: dict with s,a,ret,adv,mu,std
      info:  dict with logged mq/mx/mt/t (from tracker)
    """
    seconds = cfg.seconds
    # choose start t0 (optional phase randomization)
    if cfg.phase_random:
        slack = max(0.0, tracker.t_total - seconds)
        t0 = float(rng.uniform(0.0, slack)) if slack > 1e-6 else 0.0
    else:
        t0 = 0.0
    tracker.reset_to_time(t0)

    # roll
    t = t0
    k = 0
    S_list=[]; A_list=[]; R_list=[]; LOGP=[]; MU=[]; ENT=[]

    last_a = np.zeros(4, dtype=float)
    last_mu = np.zeros(4, dtype=float)
    last_std = np.exp(policy.log_std)

    while (t - t0) < seconds and t < tracker.t_total:
        alpha_p, alpha_d, beta_x, beta_v = gp.values()
        tracker.alpha_p = alpha_p; tracker.alpha_d = alpha_d
        tracker.beta_x  = beta_x;  tracker.beta_v  = beta_v
        
        mq, mx, mt = tracker.step_once(t)
        tau3 = float(np.linalg.norm(tracker._last_tau[:3]))
        phase = (t - t0) / max(1e-6, seconds)
        s = _state_vec(mq, mx, mt, phase, gp, scales)
        
        if (k % max(1, cfg.action_stride)) == 0:
            a, logp, ent, mu, std = policy.act(s, rng)
            gp.step(a, frac_step=cfg.frac_step)
            last_a, last_mu, last_std = a, mu, std
        else:
            a, logp, ent, mu, std = last_a, 0.0, 0.0, last_mu, last_std
            
        r = -(rews.WQ*mq + rews.WX*mx + rews.WT*mt) - rews.WLIM*np.mean(a*a) - rews.WT3*tau3 + rews.WENT*ent
        S_list.append(s); A_list.append(a); R_list.append(float(r))
        LOGP.append(float(logp)); MU.append(mu); ENT.append(float(ent))

        t += tracker.dt_sim; k += 1
    
    S_arr = np.array(S_list)
    A_arr = np.array(A_list)
    R_arr = np.array(R_list, dtype=float)
    N = len(R_arr)
    ret = np.zeros(N, dtype=float)
    running = 0.0
    for i in range(N-1, -1, -1):
        running = R_arr[i] + cfg.gamma * running
        ret[i] = running
    V = np.array([policy.value(s) for s in S_arr], dtype=float)
    adv = ret - V

    batch = {
        "s": S_arr, "a": A_arr, "ret": ret, "adv": adv,
        "mu": np.array(MU), "std": np.exp(policy.log_std), "logp": np.array(LOGP),
    }
    info = {
        "mq": getattr(tracker, "_log_mq", []).copy(),
        "mx": getattr(tracker, "_log_mx", []).copy(),
        "mt": getattr(tracker, "_log_mt", []).copy(),
        "t":  getattr(tracker, "_log_t",  []).copy(),
    }
    return batch, info