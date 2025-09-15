# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:47:05 2025

@author: YAKE
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import csv
import numpy as np

from utils import sample_ref
import matplotlib.pyplot as plt


class MetricsLogger:
    """
    Records per-step:
      - mq: joint-space RMS error
      - mx: task-space position error (configurable aggregation)
      - mt: mean absolute torque
    Then prints a summary and optional CSV, and suggests SCLs from chosen percentile.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        # weights in J = -(wq*mq + wx*mx + wt*mt)
        self.wq  = float(cfg.get("wq", 1.0))
        self.wx  = float(cfg.get("wx", 0.5))
        self.wt  = float(cfg.get("wt", 1e-4))

        # mx aggregation across tasks: mean | rms | max
        self.mx_agg = str(cfg.get("task_error_agg", "mean")).lower()
        self.mx_weights = cfg.get("task_error_weights", None)

        # output
        self.write_csv = bool(cfg.get("write_csv", False))
        self.csv_path  = str(cfg.get("csv_path", "metrics_no_rl.csv"))
        
        self.plot = bool(cfg.get("plot", True))
        self.fig_path = cfg.get("fig_path", "metrics_plot.png")
        
        self.scl_percentile = float(cfg.get("scl_hint_percentile", 90.0))

        # cache
        self.t_list:         list[float] = []
        self.mq_list:        list[float] = []
        self.mx_list:        list[float] = []
        self.mt_list:        list[float] = []
        self.tau3norm_list:  list[float] = []


    # ----- per-step update -----
    def update(self, tracker, 
               t: float, tau: np.ndarray,
               mq: float | None = None,
               mx: float | None = None,
               mt: float | None = None):
        if mq is None:
            q_ref = sample_ref(tracker.qpos_ref, t, tracker.REF_DT, tracker.t_total)
            e = q_ref - tracker.data.qpos
            mq = float(np.linalg.norm(e) / np.sqrt(max(1, e.size)))
        if mx is None:
            mx = self._compute_task_error(tracker, t)
        if mt is None:
            mt = float(np.mean(np.abs(tau)))
        tau3_norm = float(np.linalg.norm(tau[:3])) if tau.size >= 3 else float(np.linalg.norm(tau))
    
        self.t_list.append(float(t))
        self.mq_list.append(float(mq))
        self.mx_list.append(float(mx))
        self.mt_list.append(float(mt))
        self.tau3norm_list.append(tau3_norm)

    # ----- task-space aggregation -----
    def _compute_task_error(self, tracker, t: float) -> float:
        if (tracker.xpos_ref is None) or (len(tracker.task_ids) == 0):
            return 0.0
        x_d = sample_ref(tracker.xpos_ref, t, tracker.REF_DT, tracker.t_total)  # (S,3)
        if x_d is None:
            return 0.0
        # current positions per task
        X_now = []
        for tid in tracker.task_ids:
            if tracker.task_mode == 'site':
                X_now.append(tracker.data.site_xpos[tid])
            else:
                X_now.append(tracker.data.xpos[tid])
        X_now = np.asarray(X_now)                # (S,3)
        norms = np.linalg.norm(x_d - X_now, axis=1)  # (S,)

        # optional weighting per task
        if isinstance(self.mx_weights, (list, tuple, np.ndarray)) and len(self.mx_weights) == norms.size:
            w = np.asarray(self.mx_weights, float); w = w / (w.sum() + 1e-12)
            return float(np.dot(w, norms))

        # aggregation
        if self.mx_agg == "rms":
            return float(np.sqrt((norms**2).mean()))
        elif self.mx_agg == "max":
            return float(norms.max())
        else:  # mean
            return float(norms.mean())

    # ----- summary & CSV -----
    def summarize_and_maybe_write_csv(self):
        t  = np.asarray(self.t_list, float)
        mq = np.asarray(self.mq_list, float)
        mx = np.asarray(self.mx_list, float)
        mt = np.asarray(self.mt_list, float)
        tn = np.asarray(self.tau3norm_list, float)

        def stats(a):
            if a.size == 0:  # avoid nan
                return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
            return {
                "mean":   float(a.mean()),
                "median": float(np.median(a)),
                "p90":    float(np.percentile(a, 90)),
                "p95":    float(np.percentile(a, 95)),
                "max":    float(a.max()),
            }

        S_mq, S_mx, S_mt, S_tn = stats(mq), stats(mx), stats(mt), stats(tn)
        J_mean = - (self.wq*S_mq["mean"] + self.wx*S_mx["mean"] + self.wt*S_mt["mean"])

        print("\n===== Tracking Metrics =====")
        print(f"mq  (joint RMS): {S_mq}")
        print(f"mx  (task {self.mx_agg}): {S_mx}")
        print(f"mt  (mean |tau|): {S_mt}")
        print(f"tn  (residual force): {S_tn}")
        print(f"J_mean = - (wq*mq_mean + wx*mx_mean + wt*mt_mean) = {J_mean:.6f} "
              f"(wq={self.wq}, wx={self.wx}, wt={self.wt})")

        pct = int(self.scl_percentile)
        def pct_of(a):
            if a.size == 0: return 0.0
            return float(np.percentile(a, pct))
        eq_scl = pct_of(mq); ex_scl = pct_of(mx); mt_scl = pct_of(mt)
        print(f"Suggested SCL (≈ p{pct}):  EQ_SCL={eq_scl:.6g},  EX_SCL={ex_scl:.6g},  MT_SCL={mt_scl:.6g}")

        if self.write_csv:
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "mq", "mx", "mt"])
                    for ti, mqi, mxi, mti in zip(self.t_list, self.mq_list, self.mx_list, self.mt_list):
                        w.writerow([ti, mqi, mxi, mti])
                print(f"Wrote {self.csv_path}")
            except Exception as e:
                print("[WARN] failed to write CSV:", e)
        
        if self.plot:
            self._plot_figure(t, mq, mx, mt, tn, self.fig_path)
    
    def _plot_figure(self, t, mq, mx, mt, tn, fig_path=None):
        if t.size == 0:
            print("[INFO] no data to plot.")
            return
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(t, mq); axs[0].set_ylabel("mq (joint RMS (radian))")
        axs[1].plot(t, mx); axs[1].set_ylabel("mx (task MEAN (m))")
        axs[2].plot(t, mt); axs[2].set_ylabel("mt (torque MEAN (Nm))")
        axs[3].plot(t, tn); axs[3].set_ylabel("||tau[:3]|| (Normal residual force (N))")
        axs[3].set_xlabel("time [s]")

        for ax in axs:
            ax.grid(True, linewidth=0.5, alpha=0.4)

        fig.suptitle("Tracking metrics")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if fig_path:
            try:
                fig.savefig(fig_path, dpi=150)
                print(f"Wrote {fig_path}")
            except Exception as e:
                print("[WARN] failed to save figure:", e)
        try:
            plt.show()
        except Exception:
            pass