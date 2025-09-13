# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:58:09 2025

@author: YAKE
"""

import mujoco
import mujoco.viewer as viewer
import pickle
import numpy as np
import time

model_path = 'RKOB_simplified_upper_with_marker.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mj_viewer = viewer.launch_passive(model, data)
mj_dt = model.opt.timestep

jnt_name = [model.joint(i).name for i in range(model.njnt)]

traj_path = 'AJ026_Run_Comfortable.pkl'
with open(traj_path, "rb") as f:
    traj = pickle.load(f)

qpos_ref = traj["mj_qpos"]
xpos_ref = traj["mj_xpos"]
Fs = 50
DT = 1.0 / 50

frames, ndofs = qpos_ref.shape
if ndofs != len(jnt_name):
    raise ValueError("The traj's dofs {ndofs} doesn't match the model's dofs {len(jnt_name)}")

for i in range(frames):
    data.qpos[:] = qpos_ref[i,:]
    mujoco.mj_fwdPosition(model, data)
    mj_viewer.sync()
    time.sleep(1/50)
    
