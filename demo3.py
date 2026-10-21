# demo3: 把摆焊波型贴到任意形状的焊缝上（TrajGenerator 演示）
#   S 型焊缝 + 半圆收尾：只给多个离散点，位置用 PCHIP 逐坐标拟合，
#   每点带 ABC 姿态（世界系下的绝对姿态角），姿态在控制点之间线性过渡
# 插补不走 sPlanner 的速度规划，直接按路径参数 u 等步长推进（参数插补）。
# 暂时不考虑摆焊轨迹自身的平滑插补（尖角、二阶连续等），
# 只做「波型局部点 → 焊缝标架」的映射。

import numpy as np

from Traj_Planner.Sine import Sine
from Traj_Planner.DataClass import WavePara
from Traj_Planner.Pchip import Pchip
from Traj_Planner.TrajGenerator import TrajGenerator
from Traj_Planner.Vis import Vis, set_3d_axis_equal


# ==============================
# 0. 参数
# ==============================
WAVE_LEN = 3.0           # 给定波长 mm（波数由 TrajGenerator 按焊缝长度反算）
AMP = 4.0               # 摆幅 mm
DT = 0.1              # 插补周期 s


def newWavePara():
    """每个 case 用独立的 WavePara：TrajGenerator 会改写 len 与 waveNum，不能共用。"""
    p = WavePara()
    p.T = 1.0
    p.len = WAVE_LEN     # 给定波长，随后由 _checkSpan 定波数并回算波长
    p.ampLeft = AMP
    p.ampRight = AMP
    p.tiltAngle = 0.0
    p.dwell_left = 0.0
    p.dwell_mid = 0.0
    p.dwell_right = 0.0
    p.dwell_end = 0.0
    p.isMovingWhenDwell = False
    return p


def maxDeviation(seamPts, pts, block=256):
    """轨迹点到焊缝中心线的最大距离（应 ≤ 摆幅）。"""
    worst = 0.0
    for i in range(0, len(pts), block):
        b = pts[i:i + block]
        d = np.linalg.norm(b[:, None, :] - seamPts[None, :, :], axis=2)
        worst = max(worst, float(d.min(axis=1).max()))
    return worst


def makeSeam(ctrlPts):
    """多个离散点 -> seam(t)，t ∈ [0, 1]：位置 PCHIP 拟合，姿态分段线性。

    ctrlPts 每行 3 个数 (x, y, z) 或 6 个数 (x, y, z, a, b, c)：
      - 位置：PCHIP 以控制点序号为参数（等间距节点），x/y/z 各自独立拟合，
        保形不过冲，曲线形状完全由控制点决定，重复点由 Pchip 内部去重；
      - 姿态：相邻控制点之间线性过渡（焊枪姿态是逐步调整的）。
    返回的 seam(t) 与输入同宽。
    """
    P = np.asarray(ctrlPts, dtype=float)
    P = P.reshape(len(ctrlPts), -1)
    t = np.linspace(0.0, 1.0, len(P))
    itp = [Pchip(t, P[:, k]) for k in range(3)]

    def seam(tt):
        tt = float(tt)
        out = [float(itp[k].eval(tt)) for k in range(3)]
        out += [float(np.interp(tt, t, P[:, k])) for k in range(3, P.shape[1])]
        return out

    return seam, P


def resampleByArc(pts, ds):
    """把折线按弧长等间距重采样（保留首末端点），用于均匀布置控制点。"""
    P = np.asarray(pts, dtype=float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    n = max(2, int(np.floor(s[-1] / ds)))
    sq = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(sq, s, P[:, k]) for k in range(3)])


def buildSSeamPts(L=140.0, H=170.0, R=None, ds=6.0, aMax=70.0):
    """S 型焊缝控制点（6 维）：S 段 + 末端半圆回折，并带 ABC 姿态。

    位置：0~L 沿 +X 推进，横向按 smoothstep 过渡（两端平缓、中点反曲的 S 形，
         中段斜率最大约 1.8，即 60°）；随后接 180° 半圆收尾。
         R 默认取 S 段端点的曲率半径 L²/(6H)，且半圆朝 -Y 弯（与 S 段末端
         同向），于是交界处切向与曲率都连续（C2），出口切向反转为 -X。
         控制点按弧长等间距 ds 取，避免疏密突变让 PCHIP 局部产生曲率尖峰。
    姿态：ABC 为欧拉角（度，R = Rz(c)·Ry(b)·Rx(a)），标定/示教时相对**世界
         坐标系**（基坐标系）给出，是绝对姿态；转到单位向量 v 后只表示焊枪
         姿态（不是 Z 轴）：工件系 Y = v0 × X、Z = X × Y，
         插补标架 N = v × T、B = T × N。这里让 A 从 0° 线性变到 aMax
         （绕世界系 X 轴逐步侧倾），B = C = 0，
         相邻控制点之间的姿态线性过渡。
    """
    if R is None:
        R = L * L / (6.0 * H)      # S 段端点曲率半径，使交界处曲率连续

    # 1) 密集采样解析形状：S 段 + 半圆段（半圆朝 -Y，接 S 段末端的弯曲方向）
    sDense = [[L * (k / 400), H * (3 * (k / 400) ** 2 - 2 * (k / 400) ** 3), 0.0]
              for k in range(401)]
    cy = H - R
    aDense = [[L + R * np.sin(np.pi * k / 400),
               cy + R * np.cos(np.pi * k / 400), 0.0] for k in range(401)]

    # 2) 两段各自按弧长等间距取点：交界点 (L, H) 必然落在控制点上
    #    （它是 y 的局部极大，PCHIP 只有把顶点放在节点上才不会把曲线拱歪）
    P = np.vstack([resampleByArc(sDense, ds), resampleByArc(aDense, ds)[1:]])

    # 3) 姿态：整条焊缝上 A 线性 0 -> aMax，相邻控制点之间即线性过渡
    n = len(P)
    return [[float(P[i, 0]), float(P[i, 1]), float(P[i, 2]),
             aMax * i / (n - 1), 0.0, 0.0] for i in range(n)]


# ==============================
# 1. 焊缝
# ==============================
CURVE_PTS = buildSSeamPts()
_curveSeam, _ = makeSeam(CURVE_PTS)

seams = [
    # (名称, seam 可调用对象, PCHIP 控制点或 None)
    ("S 型焊缝 + 半圆收尾(多点 PCHIP 拟合)", _curveSeam, CURVE_PTS),
]


# ==============================
# 2. 逐个焊缝生成轨迹
# ==============================
cases = []
for name, seam, ctrl in seams:
    print("\n" + "#" * 66)
    print(f"# {name}")
    print("#" * 66)

    # --- 2.1 建波型 + 生成器（只给波长，波数由 TrajGenerator 定） ---
    wave = Sine(newWavePara())
    gen = TrajGenerator(seam, wave, seamSamples=3000)
    print(gen.summary())
    Xb, Yb, Zb = gen.baseFrame                 # 焊缝工件坐标系（描述焊缝几何）
    print(f"  工件系 X(起→终) = {np.round(Xb, 4)}")
    print(f"  工件系 Y        = {np.round(Yb, 4)}")
    print(f"  工件系 Z        = {np.round(Zb, 4)}")
    if ctrl is not None:
        print(f"  PCHIP 控制点  = {len(ctrl)} 个（t 等间距节点）")

    # --- 2.2 参数插补：u 随时间线性推进（不做速度规划），逐点贴到焊缝 ---
    u_end = float(wave.para.waveNum)                 # 由 _checkSpan 反算得到
    T_total = float(wave.para.T) * u_end          # 与波型周期一致：总时长 = 周期 × 波数
    n = int(np.ceil(T_total / DT)) + 1
    t_arr = np.linspace(0.0, T_total, n)
    u_arr = np.linspace(0.0, u_end, n)            # 参数等步长
    trace = [(float(t), *gen.CU(float(u))) for t, u in zip(t_arr, u_arr)]
    pts = np.array([p[1:] for p in trace])

    # --- 2.3 几何校验 ---
    seamPts = gen._seamPts                       # 焊缝中心线离散点
    dev = maxDeviation(seamPts, pts)
    errStart = np.linalg.norm(gen.CU(0.0) - seamPts[0])
    errEnd = np.linalg.norm(gen.CU(u_end) - gen.seamAtS(gen.weldLen))
    print(f"  插补点数      = {len(trace)}")
    print(f"  起点贴合误差  = {errStart:.2e} mm，终点贴合误差 = {errEnd:.2e} mm")
    print(f"  到焊缝中心线最大偏移 = {dev:.4f} mm（摆幅 {AMP} mm）")
    print(f"  三维路径弧长  = {np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)):.3f} mm")

    cases.append((name, gen, pts, trace, ctrl))


# ==============================
# 3. 可视化
# ==============================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 图1：焊缝上的合成轨迹 ---
nCase = len(cases)
fig = plt.figure(figsize=(7.5, 6) if nCase == 1 else (12, 5.5))
for k, (name, gen, pts, _tr, ctrl) in enumerate(cases):
    ax = fig.add_subplot(1, nCase, k + 1, projection='3d')
    sp = gen._seamPts
    ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], 'k--', lw=1.6, label='焊缝中心线(PCHIP)')
    if ctrl is not None:
        c = np.asarray(ctrl, dtype=float)
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c='m', s=28, label='PCHIP 控制点')
        if c.shape[1] >= 6:                        # 抬枪方向 B（由姿态向量叉乘导出）
            ss = np.linspace(0.0, gen.seamLen, c.shape[0])[::3]
            P = np.array([gen.seamAtS(s) for s in ss])
            U = np.array([gen.frameAtS(s)[1] for s in ss])
            ax.quiver(P[:, 0], P[:, 1], P[:, 2],
                      U[:, 0], U[:, 1], U[:, 2],
                      length=22.0, normalize=True, color='darkorange',
                      lw=1.1, label='抬枪方向 B')
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', lw=0.7, label='摆焊轨迹')
    ax.scatter(*pts[0], c='g', s=45, label='起点')
    ax.scatter(*pts[-1], c='r', s=45, label='终点')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{name}  L={gen.seamLen:.1f} mm')
    ax.legend(fontsize='small')
    ax.grid(True)
    set_3d_axis_equal(ax)
fig.suptitle('Sine 波型贴到 S 型焊缝（末端半圆收尾）上', fontsize=14, fontweight='bold')
plt.tight_layout()

# --- 图2：插补结果（无速度规划，速度只由参数步长 × |dP/du| 决定） ---
_, _gen, _pts, trace, _ctrl = cases[-1]
dt_step = trace[1][0] - trace[0][0]          # 实际时间步长
Vis.plotKinematics(trace, dt_step, plotTraj=False)   # 3D 轨迹已由图1 展示，这里只画运动学

plt.show()