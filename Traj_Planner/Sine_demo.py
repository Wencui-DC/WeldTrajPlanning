from Sine import Sine
from Numeric import *
from DataClass import *


# 创建波型参数
wavePara = WavePara()
wavePara.T = 1.0
wavePara.len = 5.0
wavePara.ampLeft = 5
wavePara.ampRight = 5
wavePara.tiltAngle = 0
wavePara.dwell_left = 0.2
wavePara.dwell_right = 0.2
wavePara.dwell_mid = 0
wavePara.dwell_end = 0
wavePara.isMovingWhenDwell = False


# 构建sine实例化
numWave = 3
sine = Sine(wavePara)

# 计算曲率限速
aMax = 10000 # mm/s^2
maxSpeed = sine.computeGlobalCurvSpeed(aMax)
print(f"最大限速：{maxSpeed:.2f} mm/s")

cornerInfo = sine.findCorners()
print(cornerInfo)
sine.plotCorners()

# sample_cnt = 500000
# L_ref = Numeric.calcRefArcLen(sine.CU, sample_cnt=sample_cnt) * numWave

# print(f"理论值（{sample_cnt/10000:.0f}万次采样）：{L_ref:.10f} mm")
# print(f"总弧长（高斯积分）：{sine.path.arcLen * numWave:.10f} mm")
# print(f"二者绝对误差: {abs(sine.path.arcLen * numWave - L_ref):.2e}")

# 绘图（打开 Pchip 拟合）
# sine.plot(numWave, openItp=True)





