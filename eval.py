from pathlib import Path
from typing import Dict, Any
import csv, pickle, time
import numpy as np
import mujoco

from utils import load_config
from traj_tracker import TrajectoryTracker
from rl.policy_linear import LinearGaussianPolicy
from rl.gain_param import GainParam4
from rl.rollout import rollout_episode, RolloutCfg, RolloutScales, RewardWeights


# ---------- builders ----------
def _build_tracker_from_cfg(cfg: Dict[str, Any]) -> TrajectoryTracker:
    assets = cfg.get("assets", {})
    xml_path  = assets.get("xml",  "RKOB_simplified_upper_with_marker.xml")
    traj_path = assets.get("traj", "AJ026_Run_Comfortable.pkl")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    qpos_ref = np.asarray(traj["mj_qpos"], float)

    tracker = TrajectoryTracker(model, data, qpos_ref, cfg)
    return tracker


def _load_policy(policy_path: str, state_dim=10, action_dim=4) -> LinearGaussianPolicy:
    return LinearGaussianPolicy.load_npz(policy_path, state_dim=state_dim, action_dim=action_dim)


def _cfg_to_objs(cfg: Dict[str, Any]):
    rl = cfg.get("rl", {})
    ro_cfg = RolloutCfg(
        seconds=float(rl.get("seconds", 3.0)),
        gamma=float(rl.get("gamma", 0.99)),
        action_stride=int(rl.get("action_stride", 5)),
        frac_step=float(rl.get("frac_step", 0.03)),
        phase_random=False,
    )
    sc = rl.get("scales", {})
    scales = RolloutScales(
        EQ_SCL=float(sc.get("EQ_SCL", 0.1)),
        EX_SCL=float(sc.get("EX_SCL", 0.03)),
        MT_SCL=float(sc.get("MT_SCL", 60.0)),
        TAU3_SCL=float(sc.get("TAU3_SCL", 200.0)),
    )
    rw = rl.get("reward_weights", {})
    rews = RewardWeights(
        WQ=float(rw.get("WQ", 1.0)),
        WX=float(rw.get("WX", 1.0)),
        WT=float(rw.get("WT", 5e-4)),
        WT3=float(rw.get("WT3", 0.5)),
        WLIM=float(rw.get("WLIM", 1e-3)),
        WENT=float(rw.get("WENT", 1e-3)),
    )
    gb = rl.get("gain_bounds", {})
    lo = np.array(gb.get("lo", [0.2, 0.2, 0.2, 0.2]), float)
    hi = np.array(gb.get("hi", [5.0, 5.0, 5.0, 5.0]), float)
    return ro_cfg, scales, rews, lo, hi


# ---------- viewer step helper ----------
def _step_with_policy_once(tracker, t, gp, policy, rng, ro_cfg, scales, last):
    """单步推进（用于 viewer 回放），返回 mq, mx, mt, last。"""
    a_p, a_d, b_x, b_v = gp.values()
    tracker.alpha_p = a_p; tracker.alpha_d = a_d
    tracker.beta_x  = b_x; tracker.beta_v  = b_v
    
    mq, mx, mt = tracker.step_once(t)
    
    phase = (t - last["t0"]) / max(1e-6, ro_cfg.seconds)
    s = np.array([
        mq/scales.EQ_SCL, mx/scales.EX_SCL, mt/scales.MT_SCL,
        phase, np.sin(2*np.pi*phase), np.cos(2*np.pi*phase),
        *gp.normalized()
    ], dtype=float)
    
    if (last["k"] % max(1, ro_cfg.action_stride)) == 0:
        a, logp, ent, mu, std = policy.act(s, rng)
        gp.step(a, frac_step=ro_cfg.frac_step)
        last.update(a=a, logp=float(logp), ent=float(ent), mu=mu, std=std)

    last["k"] += 1
    return mq, mx, mt, last


# ---------- main ----------
def main(cfg_path: str = "cfg.yaml"):
    cfg = load_config(cfg_path)
    rl = cfg.get("rl", {})
    ev = cfg.get("eval", {})
    
    tracker = _build_tracker_from_cfg(cfg)
    ro_cfg, scales, rews, lo, hi = _cfg_to_objs(cfg)
    policy_path = ev.get("policy_path", rl.get("policy_path", "best_policy.npz"))
    policy = _load_policy(policy_path, state_dim=10, action_dim=4)
    
    seed = int(rl.get("seed", 0)) + 123  # 与训练区分
    rng = np.random.default_rng(seed)
    gp = GainParam4(lo=lo, hi=hi); gp.reset_mid()
    
    show_viewer = bool(ev.get("show_viewer", False))

    if not show_viewer:
        # ---------- 离线评测 ----------
        tracker.reset_to_time(0.0)
        batch, info = rollout_episode(tracker, gp, policy, rng, ro_cfg, scales, rews)

        with open("rl_eval_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["t", "mq", "mx", "mt"])
            T = min(len(info["t"]), len(info["mq"]), len(info["mx"]), len(info["mt"]))
            for i in range(T):
                w.writerow([info["t"][i], info["mq"][i], info["mx"][i], info["mt"][i]])

        # 写摘要
        mq_avg = float(np.mean(info["mq"])) if info["mq"] else 0.0
        mx_avg = float(np.mean(info["mx"])) if info["mx"] else 0.0
        mt_avg = float(np.mean(info["mt"])) if info["mt"] else 0.0
        J_proxy = - (rews.WQ * mq_avg + rews.WX * mx_avg + rews.WT * mt_avg)
        with open("rl_eval_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"mq_avg={mq_avg:.6e}\n")
            f.write(f"mx_avg={mx_avg:.6e}\n")
            f.write(f"mt_avg={mt_avg:.6e}\n")
            f.write(f"J_proxy={J_proxy:.6e}\n")

        print("[EVAL] wrote rl_eval_metrics.csv, rl_eval_summary.txt")
        return

    # ---------- MuJoCo 可视化 ----------
    import mujoco.viewer as viewer
    tracker.reset_to_time(0.0)
    last = {"a": np.zeros(4, float), "mu": np.zeros(4, float), "std": np.exp(policy.log_std),
            "logp": 0.0, "ent": 0.0, "k": 0, "t0": 0.0}
    t = 0.0
    with viewer.launch_passive(tracker.model, tracker.data) as v:
        while t < min(ro_cfg.seconds, tracker.t_total):
            if not v.is_running(): break
            _mq, _mx, _mt, last = _step_with_policy_once(tracker, t, gp, policy, rng, ro_cfg, scales, last)
            v.sync()
            time.sleep(1/100)
            t += tracker.dt_sim
    print("[EVAL] viewer done.")


if __name__ == "__main__":
    main("cfg.yaml")