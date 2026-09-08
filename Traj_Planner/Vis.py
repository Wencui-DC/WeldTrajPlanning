import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from WaveBase import WaveBase
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from DataClass import *

try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def set_3d_axis_equal(ax):
    """模仿 MATLAB axis equal 3D 真实等轴"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2.0
    x_mid = (xlim[0] + xlim[1]) / 2.0
    y_mid = (ylim[0] + ylim[1]) / 2.0
    z_mid = (zlim[0] + zlim[1]) / 2.0
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

class Vis(WaveBase):
    def __init__(self, wavePara:WavePara):
        super().__init__(wavePara)

    def plot(self, totalTime, openItp=False):
        cp = np.vstack([[0,0,0], self.CP])
        x = cp[:, 0]
        y = cp[:, 1]
        z = cp[:, 2]
        
        time = np.concatenate([[0], self.TList]) / self.para.T

        tq = np.linspace(0, 1, 200)

        xq, yq, zq = None, None, None
        if openItp:
            points = [self.eval(t) for t in tq]
            xq, yq, zq = zip(*points)

        # 图1 时序图 保持不变
        plt.figure(figsize=(9, 6))
        plt.subplot(3,1,1)
        plt.plot(time, x, 'bo--', lw=1.8, ms=5, label='theoretical points')
        if openItp:
            plt.plot(tq, xq, 'r-', lw=1.8, label='Pchip Interpolation')
        plt.xlabel('t / T')
        plt.ylabel('X (mm)')
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(time, y, 'mo--', lw=1.8, ms=5, label='theoretical points')
        if openItp:
            plt.plot(tq, yq, 'c-', lw=1.8, label='Pchip Interpolation')
        plt.xlabel('t / T')
        plt.ylabel('Y (mm)')
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(time, z, 'ko--', lw=1.8, ms=5, label='theoretical points')
        if openItp:
            plt.plot(tq, zq, 'b-', lw=1.8, label='Pchip Interpolation')
        plt.xlabel('t / T')
        plt.ylabel('Z (mm)')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Pchip Numerical Interpolation, v={self.v:.2f}')
        plt.tight_layout()

        # 图2 单周期3D + 真实等轴
        plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, 'ro--', lw=1.5, ms=6, label='theoretical trajectory')
        if openItp:
            ax.plot3D(xq, yq, zq, 'b-', lw=2, label='Pchip smoothed trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Space Interpolation Trajectory Comparison')
        ax.grid(True)
        ax.legend()
        set_3d_axis_equal(ax)   # 真正等轴

        # 图3 多周期完整轨迹 + 按子周期着色 + 真实等轴
        plt.figure(figsize=(8, 6))
        ax3 = plt.axes(projection='3d')
        t_full = np.linspace(0, totalTime, 2000)
        nSeg = len(self.tList)          # 每个大周期内的子段数 (8)
        normalized_TList = self.TList / self.para.T
        colors = plt.cm.tab10(np.linspace(0, 1, nSeg))  # 10色 colormap 取前 nSeg 色

        for i in range(nSeg):
            seg_x, seg_y, seg_z = [], [], []
            prev_in_seg = False
            for t in t_full:
                t_in_period = t % 1 
                # 判断该时刻属于第几个子段
                seg_idx = nSeg - 1
                for k in range(nSeg):
                    if t_in_period <= normalized_TList[k]:
                        seg_idx = k
                        break
                if seg_idx == i:
                    xt, yt, zt = self.eval(t)
                    seg_x.append(xt)
                    seg_y.append(yt)
                    seg_z.append(zt)
                    prev_in_seg = True
                else:
                    if prev_in_seg and seg_x:
                        # 插入 nan 断线,阻止跨周期/跨段连线
                        seg_x.append(np.nan)
                        seg_y.append(np.nan)
                        seg_z.append(np.nan)
                    prev_in_seg = False
            if len(seg_x) > 0:
                line, = ax3.plot3D(seg_x, seg_y, seg_z, '-', color=colors[i],
                                   lw=2)
                line.set_label(f'Seg {i+1}')

        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        ax3.set_title(f'Full Multi-Cycle Trajectory | Cycles = {totalTime}')
        ax3.grid(True)
        ax3.legend(fontsize='small', ncol=2)
        set_3d_axis_equal(ax3)  # 等轴

        plt.show()

    def plotCorners(self):
        """绘制拐点信息，将尖角位置叠加在PCHIP插补图上"""

        cornerInfo = self.corners
        cp = np.vstack([[0,0,0], self.CP])
        x = cp[:, 0]
        y = cp[:, 1]
        z = cp[:, 2]
        
        tq = np.linspace(0, 1, 200)

        points = [self.eval(t) for t in tq]
        xq, yq, zq = zip(*points)

        # 1. 提取尖角数据
        x_corners = [c['pos'][0] for c in cornerInfo]
        y_corners = [c['pos'][1] for c in cornerInfo]
        z_corners = [c['pos'][2] for c in cornerInfo]

        # 2. 生成底图 (复用 plot 方法)
        plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, 'r--', lw=1.5, ms=6, label='theoretical trajectory')
        ax.plot3D(xq, yq, zq, 'b-', lw=2, label='Pchip smoothed trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory with True Corners Highlighted')
        ax.grid(True)
        ax.legend()
        set_3d_axis_equal(ax)   # 真正等轴

        # 3. 标记尖角信息
        if cornerInfo:
            ax.scatter(x_corners, y_corners, z_corners, color='green', s=50, zorder=5, label='True Corners')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(True)
            ax.legend()

        plt.show()



    @staticmethod
    def plotKinematics(tracePoints, dt, sPlan = None, vPlan = None):
        # 根据插补点 tracePoints 画出相应的图
        # tracePoints: list of (t, x, y, z, ...)
        #   一律从实际插补点坐标反算里程/速度/加速度，不依赖插补器内部记录：
        #     里程 s = Σ|Δp|（弦长累积）
        #     速度 v = |Δp| / dt
        #     加速度 a = Δv / dt
        t_arr = np.array([p[0] for p in tracePoints])
        x_arr = np.array([p[1] for p in tracePoints])
        y_arr = np.array([p[2] for p in tracePoints])
        z_arr = np.array([p[3] for p in tracePoints])

        # 从实际插补点反算（弦长近似）
        dp = np.diff(np.column_stack((x_arr, y_arr, z_arr)), axis=0)  # 相邻点位移 (n,3)
        ds_chord = np.linalg.norm(dp, axis=1)                         # 每步弦长
        v_actual = np.insert(ds_chord / dt, 0, 0.0)                   # 速度：首点补 0
        s_actual = np.insert(np.cumsum(ds_chord), 0, 0.0)             # 里程：从 0 累积

        # 加速度
        a_mag = np.diff(v_actual) / dt
        a_mag = np.insert(a_mag, 0, 0)
        a_mag[-1] = 0

        # ---- 图1: 3D 插补轨迹 ----
        fig1 = plt.figure(figsize=(10, 7))
        ax3d = fig1.add_subplot(projection='3d')
        ax3d.plot(x_arr, y_arr, z_arr, 'b.', ms=1.5, label='插补点')
        ax3d.scatter(x_arr[0], y_arr[0], z_arr[0], c='g', s=60, label='起点')
        ax3d.scatter(x_arr[-1], y_arr[-1], z_arr[-1], c='r', s=60, label='终点')
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Z (mm)')
        ax3d.set_title(f'插补轨迹图')
        ax3d.legend()
        max_range = max(np.ptp(x_arr), np.ptp(y_arr), np.ptp(z_arr)) / 2
        mid_x, mid_y, mid_z = x_arr.mean(), y_arr.mean(), z_arr.mean()
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        # ---- 图2: 弧长 / 速度 / 加速度 ----
        _, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # --- 子图1: 弧长 s(t) ---
        ax = axes2[0]
        ax.plot(t_arr, sPlan, 'r-', lw=1.5, label='累积弧长（规划）') if sPlan is not None else None
        ax.plot(t_arr, s_actual, 'b--', lw=2, label='累积弧长（真实）')
        ax.set_ylabel('弧长 s [mm]')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_title('TCP 累积里程')

        # --- 子图2: TCP 合成速率 |V| ---
        ax = axes2[1]
        ax.plot(t_arr, vPlan, 'r-', lw=1.5, label='速度（规划）') if vPlan is not None else None
        ax.plot(t_arr, v_actual, 'b--', lw=2, label='TCP速度（真实）')
        ax.set_ylabel('速度 v [mm/s]')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_title('TCP 速度')

        # --- 子图3: 合成加速度 |A| ---
        ax = axes2[2]
        ax.plot(t_arr, a_mag, 'b--', lw=1.5, label='加速度')
        ax.set_ylabel('加速度 a [mm/s^2]')
        ax.set_xlabel('时间 t [s]')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_title('TCP 加速度')

        plt.tight_layout()
        plt.show()