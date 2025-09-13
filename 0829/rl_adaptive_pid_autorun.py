#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_adaptive_pid_autorun.py
一键运行版：不需要命令行参数，直接运行即可。
"""

import sys, os
import subprocess
from pathlib import Path

# === 自动设置参数 ===
XML_CANDS = [
    "RKOB_simplified_upper_with_marker.xml",
]
XML = None
for cand in XML_CANDS:
    if Path(cand).exists():
        XML = cand
        break
if XML is None:
    raise FileNotFoundError("未找到 XML 文件，请确认模型文件名")

QPOS = "qpos_hist.csv"
if not Path(QPOS).exists():
    raise FileNotFoundError("未找到 qpos_hist.csv")

XPOS = "xpos_hist.csv" if Path("xpos_hist.csv").exists() else ""
USE_XREF = bool(XPOS)

SECONDS = 6.0
TRAIN_EPISODES = 200
DO_EVAL = False   # <<== 如果想只评估，把这里改成 True

# === 拼接成命令行参数，交给 rl_adaptive_pid_pg.py ===
args = [
    "python", "rl_adaptive_pid_pg.py",
    "--xml", XML,
    "--qpos", QPOS,
    "--seconds", str(SECONDS),
    "--train_episodes", str(TRAIN_EPISODES)
]
if USE_XREF:
    args += ["--xpos", XPOS, "--use_xref"]
if DO_EVAL:
    args.append("--eval")

print("[INFO] Running:", " ".join(args))
subprocess.run(args)
