from Traj_Planner.Sine import Sine
from Traj_Planner.DataClass import *
from Traj_Planner.Vis import *
from Numeric import *
from sPlanner import *

from geomdl import NURBS
from geomdl import utilities
from geomdl.visualization import VisMPL as vis

from basicMethods import *
import numpy as np
def make_knots_biased(n_ctrl=9, degree=3):
    """
    生成非均匀节点矢量：
    - CP2~3 附近（T/4 峰值区）节点密集
    - CP6~7 附近（3T/4 谷值区）节点密集
    - 其余区域节点稀疏
    """
    #     对应 CP 影响区间
    # u₀:  CP0~1  (起点区)   → 粗
    # u₁:  CP1~2  (爬升区)   → 过渡
    # u₂:  CP2~3  (峰值区)   → 密！离 CP2,CP3 近
    # u₃:  CP3~4  (回落区)   → 过渡
    # u₄:  CP4~5  (谷值区)   → 密！离 CP6,CP7 近
    # u₅:  CP5~6  (恢复区)   → 过渡
    # u₆:  CP6~7  (终点区)   → 粗

    # 用累积间距控制疏密: 大权重 = 大步长 = 稀疏
    weights = np.array([2.0,   # 0→1 稀疏
                         1.0,   # 1→2 正常
                         0.2,   # 2→3 密集！← CP2,CP3
                         1.0,   # 3→4 正常
                         0.2,   # 4→5 密集！← CP6,CP7
                         1.0,   # 5→6 正常
                         2.0])  # 6→7 稀疏

    cum = np.cumsum(weights)
    cum = cum / cum[-1]  # 归一化到 [0, 1]
    interior_knots = cum[:-1]  # 6 个 → 取前 6 个

    # clamped: 两端 degree+1 重复
    knots = np.concatenate([
        np.zeros(degree + 1),          # [0,0,0,0]
        interior_knots,                # 6 个内部
        np.ones(degree + 1),           # [1,1,1,1]
    ])
    return knots  # 共 4+6+4 = 14 个

knots = make_knots_biased(9, 3)
# print(knots)


# demo3：比对NURBS和PCHIP构建轨迹的不同
# ==============================
# 1. 构建波型，基于PCHIP
# ==============================
wavePara = WavePara()
wavePara.T = 1
wavePara.len = 5
wavePara.ampLeft = 5
wavePara.ampRight = 5
wavePara.tiltAngle = 70
wavePara.dwell_left = 0.1
wavePara.dwell_right = 0.1
wavePara.dwell_mid = 0
wavePara.dwell_end = 0
wavePara.isMovingWhenDwell = True

numWave = 4
sine = Sine(wavePara)

# cornerInfo = sine.findTrueCorners()
# print(cornerInfo)

# totalTime = sine.para.T * numWave
# sine.plot(totalTime, openItp=True)

# 2. 基于NURBS构建轨迹
nurbs = NURBS.Curve()
nurbs.degree = 3
ctrlpts = np.vstack([np.zeros(3), sine.CP])
# ctrlpts = basic.uniqArr(ctrlpts)
nurbs.ctrlpts = ctrlpts
nurbs.knotvector = utilities.generate_knot_vector(nurbs.degree, nurbs.ctrlpts_size)
# nurbs.knotvector = knots

# Set the visualization component and render the curve
nurbs.vis = vis.VisCurve2D()
nurbs.render()

# 3. 比对