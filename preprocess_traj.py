# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 22:03:59 2025

@author: yake
"""

import os, pickle, numpy as np

def detrend_pelvis_ty_inplace(
        pkl_path,
        out_path=None,
        bias_mode="zero_mean",
        fit_range=None,
        backup=True):
    """
    pkl_path : 原始参考文件路径（含 mj_qpos）
    out_path : 输出路径；None 则原地覆盖（可自动备份 .bak）
    bias_mode:
        - "keep_first": 只去斜率，保持首帧高度不变（默认，最安全）
        - "zero_mean" : 去斜率后再减均值，使整个轨迹在 0 周围波动
        - "match_mean": 去斜率后再把均值移回原始均值
    fit_range: (i0, i1) 可选，仅用这段帧做斜率拟合（闭区间左、开区间右，e.g. (int(T*0.1), int(T*0.9))）
    backup   : True 时，在原地覆盖时先写一个 .bak 备份
    """
    # 1) 读文件
    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)

    assert "mj_qpos" in traj, "pkl 里找不到键 'mj_qpos'"
    qpos = np.asarray(traj["mj_qpos"], float)
    T, nq = qpos.shape
    assert nq >= 3, "nq < 3，无 pelvis_ty 列"
    ty = qpos[:, 2].copy()

    # 2) 选择拟合区间
    if fit_range is None:
        i0, i1 = 0, T
    else:
        i0, i1 = int(fit_range[0]), int(fit_range[1])
        i0 = max(0, min(i0, T-2))
        i1 = max(i0+1, min(i1, T))
    idx = np.arange(i0, i1)

    # 3) 线性拟合斜率（对帧号拟合；仅为去趋势，时间尺度不影响结果）
    t = np.arange(T, dtype=float)
    a, b = np.polyfit(t[idx], ty[idx], 1)  # ty ≈ a * t + b

    # 4) 去斜率（以首帧为锚，保证首帧高度不被改动）
    #    去掉 a*(t - t[0])，这样第一帧保持不变，只消除整体上升/下降趋势
    ty_detrended = ty - a * (t - t[0])

    # 5) 去 bias（按需）
    if bias_mode == "keep_first":
        # 已保持首帧不变，这里不再改偏置
        ty_new = ty_detrended
    elif bias_mode == "zero_mean":
        ty_new = ty_detrended - ty_detrended.mean()
    elif bias_mode == "match_mean":
        ty_new = ty_detrended - ty_detrended.mean() + ty.mean()
    else:
        raise ValueError("bias_mode 只能是 'keep_first' / 'zero_mean' / 'match_mean'")

    # 6) 写回并记录处理信息
    qpos_new = qpos.copy()
    qpos_new[:, 2] = ty_new
    traj["mj_qpos"] = qpos_new
    traj.setdefault("preproc_log", {})

    # 7) 保存
    if out_path is None:
        out_path = pkl_path
        if backup and (pkl_path is not None):
            bak = pkl_path + ".bak"
            if not os.path.exists(bak):
                with open(bak, "wb") as f:
                    pickle.dump(traj, f)   # 先把处理后写到 .bak?  —— 更合理是先备份原始
                # 如果你更希望备份原始文件，请改为：
                # import shutil; shutil.copy2(pkl_path, bak)
    # 默认：把处理后的轨迹存到 out_path
    with open(out_path, "wb") as f:
        pickle.dump(traj, f)

    print(f"[OK] saved: {out_path}")
    return out_path


if __name__ == "__main__":
    # 示例：原地覆盖（会保存 .bak），只去斜率、保持首帧高度不变
    in_pkl = "AJ026_Run_Comfortable.pkl"
    detrend_pelvis_ty_inplace(in_pkl, backup=True)