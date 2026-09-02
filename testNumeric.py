from Numeric import Numeric
import numpy as np

if __name__ == "__main__":
    # Vc = 5.0
    # A = 5.0
    # frq = 1
    # omega = 2 * np.pi * frq
    # def funSine(u):
    #     x = Vc * u
    #     y = A * np.sin(omega * u)
    #     return (x, y)
        


    # numOfWaves = 1000
    # endU = (1/frq) 
    # n_order = 200
    # sample_cnt = 500000
    # eps = 1e-4
    
    # L_ref = Numeric.calcRefArcLen(funSine, 0, endU * numOfWaves, sample_cnt)
    # L_gauss, iter = Numeric.calcArcLenAdaptive(funSine, 0, endU, eps) 
    # L_gauss *= numOfWaves  # 乘以波数得到总弧长
    
    
    # print(f"============================================================================")
    # print(f"{numOfWaves}个正弦波 @ 频率 = {frq:.1f}, 振幅 = {A:.1f}, 焊接速度 = {Vc:.1f}")
    # print(f"弧长计算结果: ")
    # print(f"理论值（{sample_cnt/10000:.0f}万次采样）：{L_ref:.10f}")
    # print(f"高斯积分（{iter}次迭代）：{L_gauss:.10f}, 收敛误差: {eps:.2e}")
    # print(f"二者绝对误差: {abs(L_gauss - L_ref):.2e}")
    # print(f"============================================================================")

    ## test LegGaussPair
    xi, wi = Numeric._leggauss_weight(5)
    for x, w in zip(xi, wi):
        print(f"x: {x:.6f}, w: {w:.6f}")


