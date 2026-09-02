class Pchip:
    """
    自己构造的 PCHIP 分段三次埃尔米特插值，用法：
        pchip = Pchip(x, y)  # 构造函数，输入数据点
        yq = pchip.eval(xq)  # 求值函数，输入查询点
    """
    def __init__(self, x_data, y_data):
        x_data = list(x_data)
        y_data = list(y_data)
        
        if len(x_data) != len(y_data):
            raise ValueError("x 和 y 长度必须相同")
        
        L = len(x_data)
        boolSet = [True] * L

        # 去重
        for i in range(1, L):
            dx = x_data[i] - x_data[i-1]
            dy = y_data[i] - y_data[i-1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                boolSet[i] = False

        # 过滤重复点
        self.x = [x_data[i] for i in range(L) if boolSet[i]]
        self.y = [y_data[i] for i in range(L) if boolSet[i]]
        self.n = len(self.x)

        # 计算 PCHIP 导数（核心）
        self.d = self.compute_pchip_derivatives()

    def eval(self, xq):
        """求值接口，对标 MATLAB pp.eval(xq)"""
        if isinstance(xq, (int, float)):
            xq = [xq]

        yq = []
        for x in xq:
            # 左边界
            if x <= self.x[0]:
                yq.append(self.y[0])
                continue
            # 右边界
            if x >= self.x[-1]:
                yq.append(self.y[-1])
                continue

            # 找到区间 i
            i = self._find_last_le(self.x, x)

            # 区间参数
            xi = self.x[i]
            xi1 = self.x[i+1]
            yi = self.y[i]
            yi1 = self.y[i+1]
            di = self.d[i]
            di1 = self.d[i+1]
            h = xi1 - xi
            t = (x - xi) / h

            # Hermite 基函数
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2

            # 插值公式
            val = h00 * yi + h10 * h * di + h01 * yi1 + h11 * h * di1
            yq.append(val)

        if len(yq) == 1:
            return yq[0]
        return yq

    def _segment(self, x):
        """
        返回 (区间索引 i, 区间参数 t, 段长 h) 用于某查询点 x
        边界处夹紧到首/末段（避免越界，导数在端外取 0 亦可，这里取段内极限）
        """
        if x <= self.x[0]:
            i = 0
        elif x >= self.x[-1]:
            i = self.n - 2
        else:
            i = self._find_last_le(self.x, x)

        xi = self.x[i]
        xi1 = self.x[i + 1]
        h = xi1 - xi
        t = (x - xi) / h if h != 0 else 0.0
        return i, t, h


    def eval_deriv1(self, xq):
        """一阶导数 dy/dx（PCHIP 分段三次 Hermite 解析求导）"""
        if isinstance(xq, (int, float)):
            xq = [xq]

        yq = []
        for x in xq:
            # 区间外（已夹紧到端点）导数按段内极限，端点处取该段导数即可
            i, t, h = self._segment(x)
            yi = self.y[i]
            yi1 = self.y[i + 1]
            di = self.d[i]
            di1 = self.d[i + 1]

            # Hermite 基函数一阶导（对 t）
            h00p = 6 * t**2 - 6 * t
            h10p = 3 * t**2 - 4 * t + 1
            h01p = -6 * t**2 + 6 * t
            h11p = 3 * t**2 - 2 * t

            dydt = h00p * yi + h10p * h * di + h01p * yi1 + h11p * h * di1
            dy = dydt / h if h != 0 else 0.0
            yq.append(dy)

        if len(yq) == 1:
            return yq[0]
        return yq

    def eval_deriv2(self, xq):
        """二阶导数 d²y/dx²（PCHIP 分段三次 Hermite 解析求导）"""
        if isinstance(xq, (int, float)):
            xq = [xq]

        yq = []
        for x in xq:
            i, t, h = self._segment(x)
            yi = self.y[i]
            yi1 = self.y[i + 1]
            di = self.d[i]
            di1 = self.d[i + 1]

            # Hermite 基函数二阶导（对 t）
            h00pp = 12 * t - 6
            h10pp = 6 * t - 4
            h01pp = -12 * t + 6
            h11pp = 6 * t - 2

            d2ydt2 = h00pp * yi + h10pp * h * di + h01pp * yi1 + h11pp * h * di1
            d2y = d2ydt2 / (h * h) if h != 0 else 0.0
            yq.append(d2y)

        if len(yq) == 1:
            return yq[0]
        return yq

    def compute_pchip_derivatives(self):
        """PCHIP 导数计算，完全复刻你的 MATLAB 逻辑"""
        n = self.n
        x = self.x
        y = self.y
        d = [0.0] * n

        # 区间斜率
        h = [x[i+1] - x[i] for i in range(n-1)]
        delta = [(y[i+1] - y[i]) / h[i] for i in range(n-1)]

        # 内部点
        for i in range(1, n-1):
            if delta[i-1] * delta[i] <= 0:
                d[i] = 0.0
            else:
                d[i] = 2 * delta[i-1] * delta[i] / (delta[i-1] + delta[i])

        # 端点（和你 MATLAB pchip_end 完全一样）
        def pchip_end(d1, d2, h1, h2):
            res = ((2*h1 + h2) * d1 - h1 * d2) / (h1 + h2)
            if res * d1 < 0:
                res = 0.0
            if abs(res) > 3 * abs(d1):
                res = 3 * d1
            return res

        if n >= 2:
            d[0] = pchip_end(delta[0], delta[1], h[0], h[1])
            d[-1] = pchip_end(delta[-1], delta[-2], h[-1], h[-2])

        return d

    def _find_last_le(self, arr, x):
        """找到最后一个 <= x 的索引（和 MATLAB find(...,1,'last') 一样）"""
        for i in reversed(range(len(arr))):
            if arr[i] <= x:
                return i
        return 0