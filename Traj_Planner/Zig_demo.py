from Zig import Zig
from Numeric import *
from DataClass import *

# 创建波型参数
wavePara = WavePara()
wavePara.T = 2.0
wavePara.len = 5.0
wavePara.ampLeft = 5
wavePara.ampRight = 5
wavePara.tiltAngle = 45.0
wavePara.dwell_left = 0.2
wavePara.dwell_right = 0.2
wavePara.dwell_mid = 0.
wavePara.dwell_end = 0.
wavePara.isMovingWhenDwell = True

zig = Zig(wavePara)
numWave = 4

# sample_cnt = 500000
# L_ref = Numeric.calcRefArcLen(zig.CU, sample_cnt=sample_cnt) * numWave

# print(f"理论值（{sample_cnt/10000:.0f}万次采样）：{L_ref:.10f}")
# print(f"高斯积分（{128}次迭代）：{zig.arcLen * numWave:.10f}")
# print(f"二者绝对误差: {abs(zig.arcLen * numWave - L_ref):.2e}")

# 绘图（openItp=True 打开Pchip）
# totalTime = numWave * 1
# zig.plot(totalTime, openItp=True)

# 计算曲率限速
aMax = 10000 # mm/s^2
maxSpeed = zig.computeGlobalCurvSpeed(aMax)
print(f"最大限速：{maxSpeed:.2f} mm/s")

cornerInfo = zig.findCorners()
print(cornerInfo)
zig.plotCorners()


