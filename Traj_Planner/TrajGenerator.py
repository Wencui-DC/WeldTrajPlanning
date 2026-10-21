import sys
from pathlib import Path as _FilePath
sys.path.append(str(_FilePath(__file__).parent))
sys.path.append(str(_FilePath(__file__).parent.parent))

import math

import numpy as np
from DataClass import *

WAVE_NUM_STEP = 1     # 波数对齐粒度：WaveBase 只接受整数或 .5


def abcToPoseVec(a, b, c):
    """欧拉角 A/B/C转成世界坐标系下的向量，用于姿态描述。
    """
    a, b, c = np.radians([float(a), float(b), float(c)])
    return np.array([
        np.cos(c) * np.sin(b) * np.cos(a) + np.sin(c) * np.sin(a),
        np.sin(c) * np.sin(b) * np.cos(a) - np.cos(c) * np.sin(a),
        np.cos(b) * np.cos(a),
    ])


class TrajGenerator:
    """焊缝轨迹生成器：把"已参数化的摆焊波型"铺到"任意形状的焊缝"上。

    波型对象（Sine / Zig / ...）只在自己的局部坐标系里描述摆动：
        local x —— 沿焊缝方向的推进量 (mm)
        local y —— 横向摆动量 (mm)
        local z —— 抬枪方向分量 (mm)
    本类负责建立焊缝的活动标架（切向 T / 横向 N / 抬枪 B），
    把局部坐标 (x, y, z) 映射回世界坐标系（基坐标系）下的三维点：
        P = seamAtS(x) + y * N + z * B

    坐标系约定（世界系 = 机械臂基坐标系，控制点、ABC、插补点都在它下面描述）：
      - ABC 姿态角：标定/示教时相对**世界系**给出，abcToPoseVec 直接得到
        世界系下的姿态参考向量 v，不经过工件坐标系换算。
      - 工件坐标系（焊缝局部系）：由 baseOrigin（原点 = 焊缝起点）与
        baseFrame（X 起→终、Y 起点姿态×X、Z = X×Y）构成完整位姿，
        用于描述焊缝自身几何；worldToWork / workToWorld 做点的换算。
      - 插补标架：逐点、随焊缝移动的 (T, N, B)，X 取逐点切向。

    Parameters
    ----------
    seam : callable
        焊缝形态函数，约定 t ∈ [0, 1] -> 带姿态的六维点 (x, y, z, a, b, c)。
        直线、圆弧、样条、NURBS 均可，只要是 "t → 点" 的映射；
        也接受带 .eval(t) 方法的曲线对象。缺 ABC 直接报错：姿态是建立
        工件坐标系与插补标架的前提，没有缺省姿态。
        a/b/c 为欧拉角（度，R = Rz(c)·Ry(b)·Rx(a)），标定/示教时相对**世界
        坐标系**（基坐标系）给出，是绝对姿态；转到单位向量 v 后只表示该处
        焊枪姿态（不是 Z 轴）：标架由 X 与 v 叉乘得到，即 Y = v × X、
        Z = X × Y（见 abcToPoseVec）。
        插补时 X 取逐点切向 T，于是 N = v × T、B = T × N。
    wave : WaveBase
        已设定好参数的摆焊波型实例，例如 Sine(wavePara)、Zig(wavePara)。
        只需给定波长 para.len，波数无需指定：构造时按焊缝实长
        waveNum = ceil(seamLen / 波长) 并向上对齐到 0.5，再把波长回算为
        seamLen / waveNum，使波型恰好铺满焊缝。
    seamSamples : int, optional
        焊缝离散采样点数，用于弧长参数化与标架预计算，默认 2000。
    """

    def __init__(self, seam, wave, seamSamples=5000):
        self.seam = self._wrapSeam(seam)
        self.wave = wave
        self.tRange = (0.0, 1.0)
        self.seamSamples = max(2, int(seamSamples))

        self.seamLen = 0.0       # 焊缝弧长 (mm)
        self.weldLen = 0.0       # 摆焊长度 (mm)

        # 工件坐标系（焊缝局部系）的完整位姿：原点 + 三根轴，见 _buildBaseFrame
        self.baseOrigin = None   # 原点 O = 焊缝起点（世界系坐标）
        self.baseFrame = None    # (Xb, Yb, Zb) 三根轴（世界系单位向量）

        self._buildSeam()
        self._checkSpan()

    # ============================================================
    # 构造：焊缝离散化 + 弧长参数化 + 活动标架
    # ============================================================
    @staticmethod
    def _wrapSeam(seam):
        if callable(seam):
            return seam
        if hasattr(seam, "eval"):
            return seam.eval
        raise TypeError(
            "seam 必须是 callable：t -> (x, y, z)，或带 eval(t) 的曲线对象")

    @staticmethod
    def _norm(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else np.zeros(3)

    @staticmethod
    def _asPoint(p):
        return np.asarray(p, dtype=float).reshape(-1)[:3]

    @staticmethod
    def _asPoseVec(p):
        """焊缝点的姿态参考向量 v（世界系）。

        ABC 是标定/示教时相对世界（基）坐标系给出的绝对姿态，
        故 v 直接就是世界系方向，不需要再过工件坐标系换算。
        """
        arr = np.asarray(p, dtype=float).reshape(-1)
        if arr.size < 6:
            raise ValueError("焊缝点缺少 ABC 姿态角，需要 6 个数 (x, y, z, a, b, c)")
        return abcToPoseVec(*arr[3:6])

    def _buildBaseFrame(self, pts, poses):
        """焊缝工件坐标系：X = 起点→终点，Y = 起点姿态向量 × X，Z = X × Y。

        用于描述焊缝自身的几何（worldToWork / workToWorld 做点坐标换算），
        不参与 ABC 的解释（ABC 是世界系下的绝对姿态）；
        插补时 X 方向也不取它，而是取逐点切向 T（见 _buildSeam 的 tang）。
        起点姿态向量取第一个控制点的 ABC（世界系）。
        返回 (Xb, Yb, Zb) 三个世界系单位向量。

        起点与终点重合、或起点姿态向量与弦平行时无法定出 Y 轴，直接报错。
        """
        chord = pts[-1] - pts[0]
        if np.linalg.norm(chord) < 1e-9:
            raise ValueError("焊缝起点与终点重合，工件坐标系的 X 轴（起点→终点）无法确定")
        Xb = self._norm(chord)

        v0 = self._asPoseVec(poses[0])
        Yb = np.cross(v0, Xb)
        if np.linalg.norm(Yb) < 1e-8:
            raise ValueError("起点姿态向量与焊缝弦方向平行，工件坐标系的 Y 轴无法确定")
        Yb = self._norm(Yb)
        Zb = self._norm(np.cross(Xb, Yb))
        return Xb, Yb, Zb

    def _buildSeam(self):
        ts = np.linspace(self.tRange[0], self.tRange[1], self.seamSamples)
        poses = [self.seam(t) for t in ts]
        pts = np.asarray([self._asPoint(p) for p in poses], dtype=float)

        chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(chord)))
        self.seamLen = float(s[-1])
        if self.seamLen <= 1e-12:
            raise ValueError("焊缝长度约为 0，请检查 seam 函数与 tRange")

        # 切向：中心差分后单位化（只取方向）
        tang = np.gradient(pts, axis=0)
        nrm = np.linalg.norm(tang, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        tang = tang / nrm

        # 工件坐标系（ABC 姿态角的参照系）：原点 = 焊缝起点，三轴由弦与起点姿态确定
        self.baseOrigin = pts[0].copy()
        self.baseFrame = self._buildBaseFrame(pts, poses)
        poses_v = [self._asPoseVec(p) for p in poses]   # 逐点姿态向量（世界系）

        self._seamT = ts
        self._seamPts = pts
        self._seamS = s
        self._seamN = np.zeros_like(pts)
        self._seamB = np.zeros_like(pts)
        for i, tg in enumerate(tang):
            self._seamN[i], self._seamB[i] = self._frame(tg, poses_v[i])

    def _frame(self, tang, poseVec):
        """由切向 X 与姿态向量 v 叉乘出标架：Y(=N) = v × X，Z(=B) = X × Y。

        poseVec : array-like(3)
            该处的姿态参考向量 v（不是 Z 轴本身），没有缺省值。
            v 沿切向 X 的分量会在叉乘中被自然消掉，故 Z 是 v 在 ⊥X 面内的
            分量方向 —— 摆动平面始终垂直于当前切向。
        """
        ref = np.asarray(poseVec, dtype=float).reshape(3)
        N = np.cross(ref, tang)
        if np.linalg.norm(N) < 1e-8:
            # 逐点退化（姿态向量恰好平行于该点切向）：换用与切向最不平行的坐标轴
            axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ref = max(axes, key=lambda a: float(np.linalg.norm(np.cross(a, tang))))
            N = np.cross(ref, tang)
        N = self._norm(N)
        B = self._norm(np.cross(tang, N))
        return N, B

    def _checkSpan(self):
        """按给定波长把波型铺满整条焊缝：先定波数，再回算波长。

        waveNum = ceil(焊缝弧长 / 给定波长)，向上对齐到 0.5
        （WaveBase 只接受整数或 .5 的波数）；随后把波长微调为
        seamLen / waveNum，使波型恰好铺满焊缝，不留半截波。
        """
        lam0 = float(self.wave.para.len)
        if lam0 <= 1e-12:
            raise ValueError("wavePara.len（给定波长）为 0，无法反算波数")

        num = math.ceil(self.seamLen / lam0 / WAVE_NUM_STEP) * WAVE_NUM_STEP
        waveNum = max(WAVE_NUM_STEP, num)
        lam = self.seamLen / waveNum

        self.wave.rebuild(waveNum=waveNum, waveLen=lam)
        self.weldLen = float(getattr(self.wave.path, "arcLen", 0.0))
        print(f"[TrajGenerator] 波长 {lam0:.3f} → {lam:.3f} mm，"
              f"波数 {waveNum:.1f}（焊缝 {self.seamLen:.3f} mm）")

    # ============================================================
    # 焊缝查询
    # ============================================================
    def sToT(self, s):
        """焊缝弧长 s (mm) -> 焊缝原始参数 t。"""
        return float(np.interp(float(s), self._seamS, self._seamT))

    def seamAtT(self, t):
        """焊缝原始参数 t -> 三维点。"""
        return self._asPoint(self.seam(float(t)))

    def _seamAtS(self, s):
        """焊缝弧长 s -> (点 P, 横向 N, 抬枪 B, 切向 T)。

        超出端点时沿端点切向线性延长，避免 sPlanner 终点过冲被截断。
        """
        s = float(s)
        if s < 0.0:
            N, B = self._seamN[0], self._seamB[0]
            T = np.cross(N, B)
            return self._seamPts[0] + s * T, N, B, T
        if s > self.seamLen:
            N, B = self._seamN[-1], self._seamB[-1]
            T = np.cross(N, B)
            return self._seamPts[-1] + (s - self.seamLen) * T, N, B, T

        i = int(np.searchsorted(self._seamS, s, side="right")) - 1
        i = max(0, min(i, len(self._seamS) - 2))
        ds = float(self._seamS[i + 1] - self._seamS[i])
        w = 0.0 if ds <= 1e-15 else (s - float(self._seamS[i])) / ds
        P = self._seamPts[i] + w * (self._seamPts[i + 1] - self._seamPts[i])
        N = self._norm(self._seamN[i] + w * (self._seamN[i + 1] - self._seamN[i]))
        B = self._norm(self._seamB[i] + w * (self._seamB[i + 1] - self._seamB[i]))
        return P, N, B, np.cross(N, B)

    def seamAtS(self, s):
        """焊缝弧长 s (mm) -> 焊缝上的三维点。"""
        return self._seamAtS(s)[0]

    def frameAtS(self, s):
        """焊缝弧长 s (mm) -> (横向 N, 抬枪 B, 切向 T)，均为单位向量。"""
        _, N, B, T = self._seamAtS(s)
        return N, B, T

    # ============================================================
    # 工件坐标系 <-> 世界坐标系（基坐标系）的点坐标换算
    # ============================================================
    def worldToWork(self, P):
        """世界系点 -> 工件系坐标：P_work = R_baseᵀ · (P_world − O)。

        焊缝起点在工件系为 (0, 0, 0)，终点为 (弦长, 0, 0)。
        注意：baseFrame 是 (Xb, Yb, Zb) 三元组，np.array(...) 是**行**堆叠
        = R_baseᵀ，故行向量写法下再转置一次（见下方 .T）。
        """
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        return (P - self.baseOrigin) @ np.array(self.baseFrame).T

    def workToWorld(self, P):
        """工件系坐标 -> 世界系点：P_world = O + R_base · P_work。"""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        return self.baseOrigin + P @ np.array(self.baseFrame)

    # ============================================================
    # 映射：波型局部坐标 -> 世界坐标
    # ============================================================
    def mapLocal(self, local):
        """把波型局部点 (x=沿焊缝, y=横向摆动, z=抬枪) 映射到世界坐标系。"""
        p = self._asPoint(local)
        P, N, B, _ = self._seamAtS(p[0])
        return P + p[1] * N + p[2] * B

    def CU(self, u):
        """路径参数 u ∈ [0, waveNum] -> 世界坐标系（基坐标系）下的三维点。

        u 是波型的路径参数：一个完整波形对应 u 增加 1，故 u ∈ [0, waveNum]
        正好铺满整条焊缝。返回 np.ndarray，形状 (3,)，单位 mm。
        与 WaveBase.CU 同签名，可直接喂给 sPlanner.interpolate。
        """
        return self.mapLocal(self.wave.CU(u))

    def eval(self, u):
        return self.CU(u)

    def generate(self, n=2000):
        """按 u 均匀采样整条摆焊轨迹，返回 (n, 3) 点数组。"""
        u_end = float(self.wave.para.waveNum)
        us = np.linspace(0.0, u_end, max(2, int(n)))
        return np.asarray([self.CU(u) for u in us], dtype=float)


    # ============================================================
    # 信息 / 可视化
    # ============================================================
    def summary(self):
        lines = [
            "=" * 60,
            "  焊缝轨迹生成器 TrajGenerator",
            f"  焊缝弧长      = {self.seamLen:.2f} mm",
            f"  工件系原点 O  = ({self.baseOrigin[0]:.2f}, "
            f"{self.baseOrigin[1]:.2f}, {self.baseOrigin[2]:.2f}) mm",
            f"  波型推进长度  = {self.weldLen:.2f} mm",
            f"  波数 waveNum  = {float(self.wave.para.waveNum):.2f}",
            f"  单波波长      = {float(self.wave.para.len):.2f} mm"
            f"    周期 T = {float(self.wave.para.T):.2f} s",
            f"  摆幅 左/右    = {float(self.wave.para.ampLeft):.2f} / "
            f"{float(self.wave.para.ampRight):.2f} mm",
        ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot(self, n=2000):
        """绘制焊缝中心线与叠加摆动后的三维轨迹。"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            raise ImportError("需要 matplotlib，请 pip install matplotlib")

        try:
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        matplotlib.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        traj = self.generate(n)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.plot(self._seamPts[:, 0], self._seamPts[:, 1], self._seamPts[:, 2],
                'k--', lw=1.5, label='焊缝中心线')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                'b-', lw=0.8, label='摆焊轨迹')
        ax.scatter(*traj[0], c='g', s=50, label='起点')
        ax.scatter(*traj[-1], c='r', s=50, label='终点')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('焊缝 + 摆焊波型合成轨迹')
        ax.legend()
        ax.grid(True)

        rng = max(np.ptp(traj[:, 0]), np.ptp(traj[:, 1]), np.ptp(traj[:, 2])) / 2
        mid = traj.mean(axis=0)
        if rng > 1e-9:
            ax.set_xlim(mid[0] - rng, mid[0] + rng)
            ax.set_ylim(mid[1] - rng, mid[1] + rng)
            ax.set_zlim(mid[2] - rng, mid[2] + rng)
        plt.tight_layout()
        plt.show()
