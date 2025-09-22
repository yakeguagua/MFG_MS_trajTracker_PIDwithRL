# 项目名称: 肌肉骨骼模型的轨迹跟踪
目标: 基于轨迹数据, 探索***经典PID控制方法***与***强化学习参数优化方法***在人体运动模仿中的实现

---
## 使用方法
1. 不使用RL进行参数优化
   
   python traj_tracker.py     #默认使用cfg中预定义的参数
2. 使用RL进行参数

   -- 训练: python train.py
   
   -- 测试: python eval.py

## 控制律
1. 关节空间

   qdd_cmd = qdd_ref + kp_q * (q_d - q) + kd_q * (qd_d - qd)
   
2. 任务空间

   tau_task = sum( jacp.T @ (Kx @ ex + Dx @ ev))
   
   where:
   ex = (x_d[s] - x),  ev = (v_d[s] - v)
   
   if orientation enabled:
   
   tau_task += jacr.T @ (KR @ eR + DR @ ew)

## 仓库结构
|-- 0829/                       # materials from Wu  
|-- debug/                      # Initial attempt：PID 控制 & RL 调整 PID 参数  
|-- rl/                         # 强化学习相关模块  
|   |-- gain_param.py           # 增益参数模块  
|   |-- policy_linear.py        # 线性策略定义  
|   |-- rollout.py              # rollout 逻辑  
|-- traj_data/                  # 存放轨迹数据（pkl 文件）  
|
|-- RKOB_simplified_upper_with_marker.xml  # MuJoCo 人体上肢模型 XML  
|-- best_policy.npz             # 保存的 RL 策略参数   
|-- cfg.yaml                    # 配置文件  
|  
|-- eval.py                     # 评估程序（离线测试 & 可视化）  
|-- train.py                    # 训练脚本（RL）  
|-- traj_tracker.py             # 轨迹跟踪器实现  
|-- preprocess_traj.py          # 轨迹数据预处理  
|-- utils.py                    # 工具函数  
|  
|-- metrics.py                  # 日志与评估指标  
|-- metrics_baseline.csv        # 评估基线结果  
|-- metrics_plot.png            # 可视化结果  
|-- rl_train_log.csv            # RL 训练日志  
|-- train_state.json            # RL 训练状态保存  
|  
|-- .gitignore
