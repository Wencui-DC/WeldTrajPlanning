import time
from scipy.interpolate import PchipInterpolator
from Pchip import Pchip 

# 测试数据
x = [0,1,2,3,4]
y = [0,2,1,3,1]
test_points = [0.5, 1.5, 2.5, 3.5] * 5000  # 放大循环，让时间更明显

# ======================
# 测试 pp（你的版本：带去重）
# ======================
t1 = time.time()
pp = Pchip(x, y)
for xq in test_points:
    yq = pp.eval(xq)
t_pp = time.time() - t1

# ======================
# 测试 pp2（极简无去重）
# ======================

class Pchip2:
    def __init__(self, x, y):
        self._interp = PchipInterpolator(x, y, extrapolate=True)
    def eval(self, xq):
        return float(self._interp(xq))
    
t1 = time.time()
pp2 = Pchip2(x, y)
for xq in test_points:
    yq = pp2.eval(xq)
t_pp2 = time.time() - t1

# ======================
# 打印结果
# ======================
print("\n===== 运行时间对比 =====")
print(f"pp（带去重）耗时：{t_pp:.4f} 秒")
print(f"pp2（极简版）耗时：{t_pp2:.4f} 秒")
print(f"速度提升：{t_pp / t_pp2:.1f} 倍")