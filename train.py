# train_rl.py
# -*- coding: utf-8 -*-
from pathlib import Path
import json, os, csv
import numpy as np
import mujoco
import pickle
from typing import Dict, Any

from utils import load_config
from traj_tracker import TrajectoryTracker
from rl.policy_linear import LinearGaussianPolicy, PolicyConfig
from rl.gain_param import GainParam4
from rl.rollout import rollout_episode, RolloutCfg, RolloutScales, RewardWeights


# ---------- helpers ----------
def _build_tracker_from_cfg(cfg: Dict[str, Any]) -> TrajectoryTracker:
    assets = cfg.get("assets", {})
    xml_path  = assets.get("xml")
    if xml_path is None or not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    traj_path = assets.get("traj")
    if traj_path is None or not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    qpos_ref = np.asarray(traj["mj_qpos"], float)
    tracker = TrajectoryTracker(model, data, qpos_ref, cfg)
    return tracker


def _get_scales_from_cfg(cfg: Dict[str, Any], tracker: TrajectoryTracker):
    rl_cfg = cfg.get("rl", {})

    sc = rl_cfg.get("scales", None)
    if isinstance(sc, dict) and all(k in sc for k in ("EQ_SCL", "EX_SCL", "MT_SCL")):
        EQ, EX, MT = float(sc["EQ_SCL"]), float(sc["EX_SCL"]), float(sc["MT_SCL"])
    else:
        EQ, EX, MT = 0.1, 0.03, 60.0

    TAU3 = float(rl_cfg.get("scales", {}).get("TAU3_SCL", 200.0)) if isinstance(rl_cfg.get("scales", {}), dict) else 200.0
    return RolloutScales(EQ_SCL=EQ, EX_SCL=EX, MT_SCL=MT, TAU3_SCL=TAU3)


def _make_policy_from_cfg(state_dim: int, action_dim: int, rl_cfg: Dict[str, Any], seed: int):
    pol_cfg = PolicyConfig(
        lr_pi=float(rl_cfg.get("lr", 1e-3)),
        lr_v=float(rl_cfg.get("lr", 1e-3)),
        ent_coef=float(rl_cfg.get("ent_coef", 5e-4)),
        init_std=float(rl_cfg.get("init_std", 0.08)),
        learn_std=bool(rl_cfg.get("learn_std", False)),
        ridge_reg=float(rl_cfg.get("ridge_reg", 1e-4)),
    )
    return LinearGaussianPolicy(state_dim=state_dim, action_dim=action_dim, seed=seed, cfg=pol_cfg)


def _reward_weights_from_cfg(rl_cfg: Dict[str, Any]) -> RewardWeights:
    rw = rl_cfg.get("reward_weights", {})
    return RewardWeights(
        WQ=float(rw.get("WQ", 1.0)),
        WX=float(rw.get("WX", 1.0)),
        WT=float(rw.get("WT", 5e-4)),
        WT3=float(rw.get("WT3", 0.5)),
        WLIM=float(rw.get("WLIM", 1e-3)),
        WENT=float(rw.get("WENT", 1e-3)),
    )


def _gain_bounds_from_cfg(rl_cfg: Dict[str, Any]):
    gb = rl_cfg.get("gain_bounds", {})
    lo = gb.get("lo", [0.2, 0.2, 0.2, 0.2])
    hi = gb.get("hi", [5.0, 5.0, 5.0, 5.0])
    return np.array(lo, float), np.array(hi, float)


# ---------- main training ----------
def main(cfg_path: str = "cfg.yaml"):
    cfg = load_config(cfg_path)
    rl = cfg.get("rl", {})
    
    tracker = _build_tracker_from_cfg(cfg)

    seed = int(rl.get("seed", 0))
    rng = np.random.default_rng(seed)

    ro_cfg = RolloutCfg(
        seconds=float(rl.get("seconds", 3.0)),
        gamma=float(rl.get("gamma", 0.99)),
        action_stride=int(rl.get("action_stride", 5)),
        frac_step=float(rl.get("frac_step", 0.03)),
        phase_random=bool(rl.get("phase_random", False)),
    )

    scales = _get_scales_from_cfg(cfg, tracker)

    rews = _reward_weights_from_cfg(rl)

    state_dim, action_dim = 10, 4
    policy = _make_policy_from_cfg(state_dim, action_dim, rl, seed)
    lo, hi = _gain_bounds_from_cfg(rl)
    gp = GainParam4(lo=lo, hi=hi)

    episodes = int(rl.get("episodes", 500))
    best_policy_path = Path(rl.get("policy_path", "best_policy.npz"))
    train_log_path   = Path(rl.get("train_log_csv", "rl_train_log.csv"))
    state_json_path  = Path(rl.get("train_state_json", "train_state.json"))

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ep", "J_proxy", "mq_mean", "mx_mean", "mt_mean"])

    best_J = -1e18
    
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    (axJ, axmq), (axmx, axmt) = axes
    lines = {}
    lines["J_proxy"], = axJ.plot([], [], label="J_proxy")
    axJ.set_xlabel("episode"); axJ.set_ylabel("J_proxy"); axJ.legend()
    lines["mq_mean"], = axmq.plot([], [], label="mq_mean")
    axmq.set_xlabel("episode"); axmq.set_ylabel("mq_mean"); axmq.legend()
    lines["mx_mean"], = axmx.plot([], [], label="mx_mean")
    axmx.set_xlabel("episode"); axmx.set_ylabel("mx_mean"); axmx.legend()
    lines["mt_mean"], = axmt.plot([], [], label="mt_mean")
    axmt.set_xlabel("episode"); axmt.set_ylabel("mt_mean"); axmt.legend()
    eps, J_list, mq_list, mx_list, mt_list = [], [], [], [], []
    
    fig.tight_layout()
    for ep in range(1, episodes + 1):
        gp.reset_mid()
        tracker.reset_to_time(0.0)

        batch, info = rollout_episode(tracker, gp, policy, rng, ro_cfg, scales, rews)
        policy.update(batch)

        mq_mean = float(np.mean(info["mq"])) if len(info["mq"]) else 0.0
        mx_mean = float(np.mean(info["mx"])) if len(info["mx"]) else 0.0
        mt_mean = float(np.mean(info["mt"])) if len(info["mt"]) else 0.0
        J_proxy = - (rews.WQ * mq_mean + rews.WX * mx_mean + rews.WT * mt_mean)
        
        eps.append(ep)
        J_list.append(J_proxy)
        mq_list.append(mq_mean)
        mx_list.append(mx_mean)
        mt_list.append(mt_mean)
        lines["J_proxy"].set_data(eps, J_list)
        lines["mq_mean"].set_data(eps, mq_list)
        lines["mx_mean"].set_data(eps, mx_list)
        lines["mt_mean"].set_data(eps, mt_list)
    
        for ax, data in zip([axJ, axmq, axmx, axmt],
                            [J_list, mq_list, mx_list, mt_list]):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.01)

        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, J_proxy, mq_mean, mx_mean, mt_mean])

        if J_proxy > best_J:
            best_J = J_proxy
            policy.save_npz(best_policy_path)

        print(f"[TRAIN] ep={ep:04d}  Jp={J_proxy:.4e}  mq={mq_mean:.3e}  mx={mx_mean:.3e}  mt={mt_mean:.3e}")

    state = {
        "best_J_proxy": float(best_J),
        "episodes": episodes,
        "seed": seed,
        "scales": {"EQ_SCL": scales.EQ_SCL, "EX_SCL": scales.EX_SCL, "MT_SCL": scales.MT_SCL, "TAU3_SCL": scales.TAU3_SCL},
        "rollout_cfg": {
            "seconds": ro_cfg.seconds, "gamma": ro_cfg.gamma,
            "action_stride": ro_cfg.action_stride, "frac_step": ro_cfg.frac_step,
            "phase_random": ro_cfg.phase_random
        },
        "reward_weights": {k: getattr(rews, k) for k in ("WQ","WX","WT","WT3","WLIM","WENT")},
        "gain_bounds": {"lo": lo.tolist(), "hi": hi.tolist()},
        "policy_paths": {"best": str(best_policy_path)},
        "train_log_csv": str(train_log_path),
    }
    with open(state_json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"[DONE] best policy → {best_policy_path}")
    print(f"[LOG]  train log → {train_log_path}")
    print(f"[META] state json → {state_json_path}")


if __name__ == "__main__":
    main("cfg.yaml")