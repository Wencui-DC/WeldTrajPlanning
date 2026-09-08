import numpy as np
from Traj_Planner.Sine import *
from Traj_Planner.Zig import *
from Traj_Planner.DataClass import *
from Traj_Planner.Vis import *
from Numeric import *
from sPlanner import *

# demo0: 基于轨迹的速度规划
# ==============================
# 1. 构建轨迹
# ==============================
wavePara = WavePara()
wavePara.T = 1
wavePara.len = 10
wavePara.ampLeft = 3
wavePara.ampRight = 3
wavePara.tiltAngle = 0
wavePara.dwell_left = 0.1
wavePara.dwell_mid = 0.1
wavePara.dwell_right = 0.1
wavePara.dwell_end = 0.1
wavePara.isMovingWhenDwell = True

numWave = 50
sine = Sine(wavePara, numWave)
# sine.printPath()

# cornerInfo = sine.findCorners()
# sine.plotCorners(cornerInfo)

# totalTime = sine.para.T * numWave
# sine.plot(totalTime, openItp=True)

# ==============================
# 2. 速度规划
# ==============================
aMax = 10000  # mm/s^2
jMax = 50000
# sine.computeGlobalCurvSpeed(aMax)
# print(f"曲率限速：{vMax:.2f} mm/s")
vMax = 50

dt = 0.004 #插补周期
sVelo = sPlanner(dt, vMax, aMax, jMax)
sVelo.plan(sine.path)
sVelo.printVeloPlans()  # 打印速度规划结果

# print(sVelo.summary())

# # # ==============================
# # # 3. 逐周期插补
# # # ==============================
tracePoints = []
simu_time = np.arange(0, sVelo.T, dt)
for t in simu_time:
    posi = sVelo.interpolate(sine, t)  # dt 缺省用构造时的 Ts
    x, y, z = posi[0], posi[1], posi[2]
    tracePoints.append((t, x, y, z))

print(f"插补完成：共 {len(tracePoints)} 个周期\n")

# ================================
# 绘制运动学图
# ================================
t_arr = np.array([p[0] for p in tracePoints])
v_plan = np.array([sVelo.v_at(t) for t in t_arr])
s_plan = np.array([sVelo.s_at(t) for t in t_arr])
sine.plotKinematics(tracePoints, dt, s_plan, v_plan)
