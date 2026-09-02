# demo1: 展示没有做速度规划的结果

from Traj_Planner.Sine import Sine
from Traj_Planner.DataClass import *
from Traj_Planner.Vis import *
from Numeric import *
from sPlanner import *

# ==============================
# 1. 构建轨迹
# ==============================
wavePara = WavePara()
wavePara.T = 1
wavePara.len = 5
wavePara.ampLeft = 5
wavePara.ampRight = 5
wavePara.tiltAngle = 0
wavePara.dwell_left = 0
wavePara.dwell_right = 0
wavePara.dwell_mid = 0
wavePara.dwell_end = 0
wavePara.isMovingWhenDwell = False

numWave = 4
totalTime = numWave * wavePara.T
sine = Sine(wavePara)


# ==============================
# 3. 逐周期插补
# ==============================
Ts = 0.005      # 插补周期
tracePoints = []
simu_time = np.arange(0, totalTime, Ts)
for t in simu_time:
    x, y, z = sine.eval(t)
    tracePoints.append((t, x, y, z))

print(f"\n插补完成：共 {len(tracePoints)} 个周期")

# ================================
# 绘制运动学图
# ================================
t_arr = np.array([p[0] for p in tracePoints])
Vis.plotKinematics(tracePoints, Ts)
