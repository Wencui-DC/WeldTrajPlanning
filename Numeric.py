import numpy as np
from numpy.polynomial.legendre import leggauss

class Numeric:
    @staticmethod
    def _leggauss_weight(n):
        """
        计算Legendre-Gauss权重和节点
        :param n: 高斯积分节点阶数
        :return: (xi, wi) 高斯节点数组, 对应权重数组
        """
        xi, wi = leggauss(n)
        return xi, wi



    @staticmethod
    def calcArcLen(func, u_start: float, u_end: float, n_order: int, diff_h: float = 1e-6) -> float:
        """
        拓展：三维参数曲线弧长 f(u) -> (x, y, z)
        """
        xi, wi = Numeric._leggauss_weight(n_order)
        a = u_start
        b = u_end
        scale = (b - a) / 2.0
        mid = (a + b) / 2.0

        total = 0.0
        for x, w in zip(xi, wi):
            u = scale * x + mid
            x_l, y_l, z_l = func(u - diff_h)
            x_r, y_r, z_r = func(u + diff_h)
            dx = (x_r - x_l) / (2 * diff_h)
            dy = (y_r - y_l) / (2 * diff_h)
            dz = (z_r - z_l) / (2 * diff_h)
            ds_du = np.sqrt(dx**2 + dy**2 + dz**2)
            total += w * ds_du
        return scale * total
    
    @staticmethod
    def calcArcLenAdaptive(
        func,
        u_start: float = 0.,
        u_end: float = 1.,
        eps: float = 1e-8,
        max_order: int = 256
    ) -> float:
        """
        自适应精度高斯积分，迭代提升阶数直到满足误差阈值
        :param func: 参数曲线 f(u) -> (x, y)
        :param u_start: 参数下限a
        :param u_end: 参数上限b
        :param eps: 相对误差收敛阈值
        :param init_order: 初始高斯阶数
        :param max_order: 最大允许阶数，防止无限迭代
        :param diff_h: 中心差分步长
        :return: (收敛弧长, 最终使用阶数)
        """

        # 第一轮计算
        init_order = 8
        order = init_order
        len_prev = Numeric.calcArcLen(func, u_start, u_end, order)
        while order < max_order:
            # 阶数翻倍提升精度
            order *= 2
            len_curr = Numeric.calcArcLen(func, u_start, u_end, order)

            # 计算相对误差
            abs_err = abs(len_curr - len_prev)
            rel_err = abs_err / (abs(len_curr) + 1e-16)  # 防除0

            if rel_err < eps:
                return len_curr

            len_prev = len_curr

        # 达到最大阶数仍未收敛，返回结果并提示
        return len_prev

    
    @staticmethod
    def calcRefArcLen(func, a = 0, b = 1, sample_cnt = 100000):
        """
        拓展：三维极密采样参考基准 f(u) -> (x, y, z)
        :param func: 参数曲线 f(u) -> (x, y, z)
        :param a: 参数下限
        :param b: 参数上限
        :param sample_cnt: 采样点数，越大越接近真实弧长
        :return: 曲线总弧长
        """
        us = np.linspace(a, b, sample_cnt)
        xs, ys, zs = [], [], []
        for u in us:
            x, y, z = func(u)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        dx = np.diff(xs)
        dy = np.diff(ys)
        dz = np.diff(zs)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sum(ds)



