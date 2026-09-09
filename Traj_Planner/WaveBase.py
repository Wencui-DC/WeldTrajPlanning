import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from Pchip import Pchip
from DataClass import *
from Numeric import Numeric


# ==============================
# 基类：WaveBase
# ==============================
class WaveBase:
    # ============================================================
    # 公有接口
    # ============================================================
    def __init__(self, wavePara: WavePara):
        self.para = wavePara
        self.path = Path()
        self.CP = np.zeros((8, 3))
        self.v = 0.0                # 焊缝上的速度 mm/s
        self.singleArcLen = 0.0     # 单波弧长
        self.vLimit = 0.0           # 曲率限制下的最大速度
        
        

        # 单波净周期 _newT：扣除 4 段 dwell 停顿后的纯摆动运动时间（秒）。
        #   停顿时 TCP 不前进（isMovingWhenDwell=False）时，它是每波实际的
        #   运动周期；正弦角频率 omega = 2π / _newT 也基于它。
        self._newT = 0.0

        # 单波净波长 _newWL = v × _newT：停顿时不前进的情况下，
        #   一个净周期内 TCP 沿焊缝方向推进的距离（毫米）。
        self._newWL = 0.0

        # tList/TList：单波内 8 段子段时间序列及其累计时刻（秒），
        # 由 computeTtList() 填充，供插补求值与分段积分等内部使用。
        self._tList = np.zeros(8)  # 单位 秒
        self._TList = np.zeros(8)  # 单位 秒
        
        # 插补器
        self._itp_x = None
        self._itp_y = None
        self._itp_z = None

        # 弧长→参数 查找表（LUT）
        # 弧长查找表 LUTs：元素统一为 (u网格, 累计弧长 s) 二元组
        #   - single（连续单段路径）：[(us, s)]，恰一项（全局 u）；
        #   - multi（dwell 停顿路径）：逐段一项（段局部 u∈[0,1]），
        #     停顿段（弧长=0）无几何，置 None 哨兵占位；
        # 由 buildArcLengthLUT 构建，sPlanner 插补统一按 LUTs 读取。
        self._LUTs = None

        # 尖角信息
        self._corners = None

    # compute new T
    def computeNewT(self):
        self._newT = self.para.T
        if self.para.isMovingWhenDwell:
            self._newT = self.para.T - (
                    self.para.dwell_left 
                    + self.para.dwell_right 
                    + self.para.dwell_mid 
                    + self.para.dwell_end)

    # compute new wave length
    def computeNewWL(self):
        self._newWL = self.v * self._newT

    # compute velocity
    def computeV(self):
        self.v = self.para.len / self.para.T
 
    # compute tList and TList
    def computeTtList(self):
        L = 8
        tempT = self.para.T - (
            self.para.dwell_left 
            + self.para.dwell_right 
            + self.para.dwell_mid 
            + self.para.dwell_end
        )

        t_1o4 = tempT / 4
        tList = [
            t_1o4, self.para.dwell_left, 
            t_1o4, self.para.dwell_mid,
            t_1o4, self.para.dwell_right, 
            t_1o4, self.para.dwell_end
        ]

        self._tList = np.array(tList)
        self._TList[0] = tList[0]
        for i in range(1, L):
            self._TList[i] = self._TList[i-1] + tList[i]

    def calcSingleArcLen(self):
        # single（连续单段）路径
        self.singleArcLen = self.path.arcLen / self.para.waveNum

    def buildPathSegments(self):
        """
        只要轨迹连续，就生成一条统一的路径段（不论 waveNum 是整数还是 .5），
        这样后续的 LUT、速度规划和插补都能共享同一套 U 轴定义：
          - isMovingWhenDwell=True：停顿时 TCP 仍在移动，轨迹始终连续；
          - isMovingWhenDwell=False 但四个停顿时间均为 0：同样不存在停顿。
        只有"停顿时间 > 0 且停顿时不再移动"时，才拆分为多段。
        """
        waveNum = self.para.waveNum
        if waveNum <= 0:
            raise ValueError("waveNum 必须大于 0")

        whole_groups = int(waveNum)
        remainder = waveNum - whole_groups
        if abs(remainder) > 1e-12 and abs(remainder - 0.5) > 1e-12:
            raise ValueError("waveNum 的小数部分只能是 0 或 0.5")

        if self._isSinglePath():
            self.path = self._buildSinglePath()
        else:
            self.path = self._buildMultiPath(whole_groups, remainder)

        self._calcEachSegmentLen(self.path.segments)

        return True

    def _buildSinglePath(self):
        """single（连续单段）：整条轨迹只生成一段统一路径（全局 u∈[0, waveNum]）。"""
        waveNum = self.para.waveNum
        total_u = float(waveNum)
        end_point = np.asarray(self.eval(total_u), dtype=float)
        segments = [Segment(
            index=0,
            start=[0.0, 0.0, 0.0],
            end=end_point.tolist(),
            start_u=0.0,
            end_u=total_u,
            duration=float(self.para.T * waveNum),
            seg_arc_len=0.0)]

        return Path(
            segments=segments,
            T=float(self.para.T * waveNum),
            len=float(self.para.len * waveNum),
            arcLen=0.0)
    

    def _buildMultiPath(self, whole_groups, remainder):
        """multi（dwell 停顿）：按停顿位置把整波 / 半波切分为多段。

        停顿子段（弧长≈0）在 sPlanner 中原地停留，不参与速度规划。
        """
        starts, ends, t_durations, durations, dwell_flags = self._getPathSegmentInputs()
        segments = []
        path_part_dwell_flags = []
        start_t = 0.0
        for cycle_num in range(whole_groups):
            start_t = self._appendSegment(
                segments,
                path_part_dwell_flags,
                starts,
                ends,
                t_durations,
                durations,
                dwell_flags,
                cycle_num,
                start_t)

        if abs(remainder - 0.5) <= 1e-12:
            start_t = self._appendSegment(
                segments,
                path_part_dwell_flags,
                starts,
                ends,
                t_durations,
                durations,
                dwell_flags,
                whole_groups,
                start_t,
                end_u=0.5,
            )

        return Path(
            segments=segments,
            T=float(self.para.T * self.para.waveNum),
            len=float(self.para.len * self.para.waveNum),
            arcLen=0.0)
    

    def printPath(self):
        """Print the segments and summary of one wave period."""
        if not self.path.segments:
            print("\n路径为空。")
            return

        print(f"\n单周期 {len(self.path.segments)} 段路径：")
        for segment in self.path.segments:
            print(
                f"{segment.index + 1}: "
                f"u=[{segment.start_u:.2f}, {segment.end_u:.2f}] "
                f"duration={segment.duration:.2f} s "
                f"arc={segment.seg_arc_len:.2f} mm"
            )
        print(
            f"\n焊缝长度：{self.path.len:.2f} mm，"
            f"焊接时间：{self.path.T:.2f} s，"
            f"波型数量：{self.para.waveNum:.1f}，"
            f"单波弧长：{self.singleArcLen:.2f} mm，"
            f"总弧长：{self.path.arcLen:.2f} mm\n"
        )


    def evalCP(self, cycleNum):
        # cycleNum: 周期数，从1开始
        direction = np.array([1, 0, 0])
        cp_points = self.CP + direction * self.para.len * (cycleNum - 1)
        return cp_points

    
    # 初始化插补器
    def initItp(self):
        cp = np.array(self.CP)
        cp = np.vstack([[0,0,0], cp])
        x = cp[:, 0]
        y = cp[:, 1]
        z = cp[:, 2]
        time = self._TList
        time = np.concatenate([[0], time])

        self._itp_x = Pchip(time, x)
        self._itp_y = Pchip(time, y)
        self._itp_z = Pchip(time, z)

    def findCorners(self):
        """
        精确判定 PCHIP 拟合轨迹中的真实尖角。
        原理：PCHIP 在 delta 符号反转时将导数 d[i] 置零。
            若某数据点处 x、y、z 三个分量的导数同时为 0，
            则轨迹在该点速度为 0 且方向剧变 → 真尖角。
        """
        corners = []

        # 每个 PCHIP 插补器已存储了导数数组 self._itp_x.d, self._itp_y.d, self._itp_z.d
        # 去重后的点数 = len(self._itp_x.x)
        n = self._itp_x.n
        for i in range(1, n - 1):  # 排除端点
            dx = self._itp_x.d[i]
            dy = self._itp_y.d[i]
            dz = self._itp_z.d[i]

            is_zero = lambda v: abs(v) < 1e-12
            if is_zero(dx) and is_zero(dy) and is_zero(dz):
                t_i = self._itp_x.x[i]
                pos = np.array([self._itp_x.y[i], self._itp_y.y[i], self._itp_z.y[i]])
                corners.append({
                    'idx': i,
                    't': np.round(t_i, 2),
                    'pos': pos,
                    'd': (dx, dy, dz),
                })

        # ---------------- 美观打印区域 ----------------
        if not corners:
            print("\n未检测到尖角。")
        else:
            print("\n" + "="*60)
            print(f"检测到 {len(corners)} 个真尖角")
            print("-" * 60)
            header = f"{'序号':<6} | {'时间':<8} | {'坐标':<24}"
            print(header)
            print("-" * 60)
            for c in corners:
                pos_str = f"[{c['pos'][0]:.2f}, {c['pos'][1]:.2f}, {c['pos'][2]:.2f}]"
                row = f"{c['idx']:<6} | {c['t']:<8.2f} | {pos_str:<24}"
                print(row)
            print("="*60 + "\n")

        self._corners = corners

        return corners


    
    def eval(self, t):
        # 参数 t 归一化
        if t < 0:
            raise ValueError("t should be >= 0!")
        
        cycleNum = np.floor(t)
        t -= cycleNum
        t *= self.para.T
        x = self._itp_x.eval(t) + cycleNum * self.para.len
        y = self._itp_y.eval(t)
        z = self._itp_z.eval(t)
        return x, y, z

    # 求值接口
    def CU(self, u):
        x, y, z = self.eval(u)
        return np.array([x, y, z])

    def CU_der1(self, u):
        # 返回一阶导数C'(u)
        u = self._wrap_u(u)
        itp_u = u * self.para.T
        x1, y1, z1 = (self._itp_x.eval_deriv1(itp_u),
            self._itp_y.eval_deriv1(itp_u),
            self._itp_z.eval_deriv1(itp_u))
        return np.array([x1, y1, z1]) * self.para.T

    def CU_der2(self, u):
        # 返回二阶导C''(u)
        u = self._wrap_u(u)
        itp_u = u * self.para.T
        x2, y2, z2 = (self._itp_x.eval_deriv2(itp_u),
                      self._itp_y.eval_deriv2(itp_u),
                      self._itp_z.eval_deriv2(itp_u))
        return np.array([x2, y2, z2]) * self.para.T**2

    def curvature(self, u):
        """
        计算参数 u 处轨迹的空间曲率 κ(u)
            κ = |P'(u) × P''(u)| / |P'(u)|^3
        需在 initItp() 之后调用。
        """
        
        cu1 = self.CU_der1(u)
        cu2 = self.CU_der2(u)

        speed = np.linalg.norm(cu1)
        cross = np.cross(cu1, cu2)

        kappa = np.linalg.norm(cross) / (speed**3 + 1e-12)
        return kappa # curvature

    def vMaxByCurvature(self, u, a_max):
        """
        保守曲率限速（单轴加速度上限 a_max，忽略法向量 N 的分担）
        直线段（κ≈0）返回 inf，表示不限速。
        """
        kappa = self.curvature(u)
        if kappa < 1e-12:
            return 1e12
        return np.sqrt(a_max / kappa)

    def computeGlobalCurvSpeed(self, a_max, n_scan=10000):
        """
        扫描整个单周期，求全局曲率限速的最小值（最急弯处的限速）。
        :param a_max: 单轴加速度上限（mm/s²）
        :param n_scan: 扫描点数
        :return: (v_global, u_min) 全局限速及其所在的 u
        """
        us = np.linspace(0, 1, n_scan, endpoint=False)
        v_min = 1e12
        # u_min = 0.0
        for u in us:
            v = self.vMaxByCurvature(u, a_max)
            if v < v_min:
                v_min = v
                # u_min = u
        self.vLimit = v_min
        return v_min

    # ============================================================
    # 弧长 LUT：预计算 s(u) 查找表，用于弧长→参数的反向映射
    #   self._LUTs 元素统一为 (u网格, 累计弧长 s)：
    #     - single：[(us, s)] 恰一项，u 为全局参数；
    #     - multi：逐段一项，u 为段局部 [0,1]，停顿段置 None。
    # ============================================================
    def buildArcLengthLUT(self):
        """弧长 LUT 入口：按路径形态分发并返回 self。"""
        if self._isSinglePath():
            return self._buildLutForSinglePath()
        else:
            return self._buildLutForMultiPath()


    # ============================================================
    # 私有实现（_ 开头，仅供类内部调用）
    # ============================================================
    def _isSinglePath(self):
        """整条轨迹是否连续
        isMovingWhenDwell=True 时，停顿时 TCP 仍在移动，轨迹始终连续。
        即使为 False，若四个停顿时间均为 0，轨迹同样没有停顿，此时也必须视为
        """
        if self.para.isMovingWhenDwell:
            return True

        return not (
            self.para.dwell_left > 1e-12
            or self.para.dwell_right > 1e-12
            or self.para.dwell_mid > 1e-12
            or self.para.dwell_end > 1e-12)

    def _getPathSegmentInputs(self):
        """Return endpoints, normalized durations, durations, and dwell flags."""
        if self._isSinglePath():
            return (
                np.array([[0.0, 0.0, 0.0]]),
                np.array([self.CP[-1]]),
                np.array([1.0]),
                np.array([self.para.T]),
                [False],
            )

        durations = self._tList.copy()
        return (
            np.vstack([[0.0, 0.0, 0.0], self.CP[:-1]]),
            self.CP.copy(),
            durations / self.para.T,
            durations,
            [False, True, False, True, False, True, False, True],
        )

    @staticmethod
    def _mergeSegment(
            segments, dwell_flags, start, end, start_t, end_t, duration,
            is_dwell):
        """Append a sub-segment, merging consecutive motion sub-segments.

        通用的合并规则：相邻两个子段之间**只要不存在真正的停顿**就合并为一段。
        停顿时长为 0 的 dwell 子段不会产生独立段（调用方已跳过），因此它既不
        切分路径、也不阻断合并。于是分段的唯一依据就是"中间有没有真停顿"，
        与整波 / 半波的分组无关：
          - dwell_end=0   → 整波末段与下一个（半）波的首段连成一段；
          - dwell_mid=0   → 波内相邻移动段连成一段；
          - dwell_left>0  → 该处切分，前后各自独立规划。
        合并后的段由 sPlanner 做一次整体速度规划，段内速度连续。
        """
        if segments and (not is_dwell) and (not dwell_flags[-1]):
            previous = segments[-1]
            previous.end = end.tolist()
            previous.end_u = end_t
            previous.duration += duration
            return

        segments.append(Segment(
            index=len(segments),
            start=start.tolist(),
            end=end.tolist(),
            start_u=start_t,
            end_u=end_t,
            duration=duration,
        ))

        dwell_flags.append(is_dwell)

    def _appendSegment(
            self, segments, dwell_flags, starts, ends, t_durations,
            durations, is_dwell, cycle_num, start_t, end_u=None):
        
        """Append one complete group or its prefix ending at end_u."""
        cycle_offset = np.array([self.para.len * cycle_num, 0.0, 0.0])
        group_u_duration = float(sum(t_durations))
        group_end_u = 1.0 if end_u is None else end_u
        for start, end, t_duration, duration, dwell in zip(
                starts, ends, t_durations, durations, is_dwell):
            local_start_u = start_t - cycle_num * group_u_duration
            local_end_u = local_start_u + float(t_duration)
            if local_start_u >= group_end_u - 1e-12:
                break

            if local_end_u > group_end_u + 1e-12:
                ratio = (group_end_u - local_start_u) / float(t_duration)
                t_duration = float(t_duration) * ratio
                duration = float(duration) * ratio
                local_end_u = group_end_u

            if t_duration <= 1e-12:
                continue

            segment_start = np.asarray(start) + cycle_offset
            segment_end = np.asarray(end) + cycle_offset
            if local_end_u < float(end_u or 1.0) - 1e-12:
                segment_end = np.asarray(end) + cycle_offset
            elif end_u is not None:
                segment_end = np.asarray(self.eval(cycle_num + local_end_u))

            group_start_t = cycle_num * group_u_duration + local_start_u
            segment_end_t = cycle_num * group_u_duration + local_end_u
            self._mergeSegment(
                segments,
                dwell_flags,
                segment_start,
                segment_end,
                group_start_t,
                segment_end_t,
                duration,
                dwell,
            )

            start_t = segment_end_t

        return start_t

    def _pchipBreakpoints(self, u_start, u_end):
        """PCHIP 节点在 (u_start, u_end) 内的 u 坐标，用于分段积分。

        |C'(u)| 在 PCHIP 节点处不光滑。若对跨越多个节点的区间做单次高斯积分，
        会“假收敛”并低估弧长（实测连续 4.5 波路径低估约 0.7%），
        导致速度规划的总长小于真实路径长度、插补走不到终点。故必须按节点切分。
        """
        local = [0.0]
        local.extend(float(t) / self.para.T for t in self._TList)

        nodes = set()
        for k in range(int(np.floor(u_start)) - 1, int(np.ceil(u_end)) + 2):
            for u_local in local:
                node = float(k + u_local)
                if u_start + 1e-12 < node < u_end - 1e-12:
                    nodes.add(node)
        return sorted(nodes)

    def _integrateArcLen(self, u_start, u_end):
        """按 PCHIP 节点切分后逐段自适应积分，保证弧长精度。"""
        bounds = [float(u_start)]
        bounds.extend(self._pchipBreakpoints(u_start, u_end))
        bounds.append(float(u_end))

        total = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a <= 1e-12:
                continue

            total += Numeric.calcArcLenAdaptive(self.CU, a, b)

        return total

    def _calcEachSegmentLen(self, path_parts):
        """Calculate each segment's arc length and return their sum."""
        for segment in path_parts:
            if np.linalg.norm(np.asarray(segment.end) - np.asarray(segment.start)) <= 1e-12:
                segment.seg_arc_len = 0.0
            else:
                segment.seg_arc_len = self._integrateArcLen(segment.start_u, segment.end_u)

        # return sum(segment.seg_arc_len for segment in path_parts)
        return True

    def _wrap_u(self, u):
        # 参数 u 归一化
        cycleNum = np.floor(u)
        u = u - cycleNum
        return u

    def _buildLutForSinglePath(self):
        """single（连续单段）：整段一条 LUT（全局 u↔s 折线）。

        每正弦波均匀加密 per_wave 点，另并入段边界、整波节点与
        TList 时刻折点，避免这些位置产生弦长插值误差。
        """
        u_start = float(self.path.segments[0].start_u)
        u_end = float(self.path.segments[-1].end_u)

        n_wave = max(1, int(np.ceil(u_end - u_start)))
        per_wave = 400
        n_samples = per_wave * n_wave
        us = np.linspace(u_start, u_end, n_samples)

        extra = [float(seg.end_u) for seg in self.path.segments[:-1]]
        extra += [float(k) for k in range(1, int(u_end) + 1)]
        for cycle in range(0, int(u_end) + 2):
            for t_knot in self._TList:
                extra.append(float(cycle + float(t_knot) / self.para.T))

        extra = [x for x in extra if u_start + 1e-12 < x < u_end - 1e-12]
        if extra:
            us = np.unique(np.concatenate([us, np.asarray(extra, dtype=float)]))

        # 弦长累积：一次向量化求值所有采样点，再逐段取欧氏距离累加
        pts = np.asarray([self.CU(u) for u in us], dtype=float)
        s = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))

        self._LUTs = [(us, s)]
        self.path.arcLen = s[-1]

        return self

    def _buildLutForMultiPath(self, seg_samples=200):
        """multi（dwell 停顿）：LUTs 逐段一项（段局部 u∈[0,1]↔弧长）。

        停顿段（弧长=0）无几何，置 None 哨兵占位。
        """
        self._LUTs = []
        for seg in self.path.segments:
            if float(seg.seg_arc_len) <= 1e-12:
                self._LUTs.append(None)
                continue

            u0 = float(seg.start_u)
            u1 = float(seg.end_u)
            span = u1 - u0
            us_l = np.linspace(0.0, 1.0, int(seg_samples))
            us_g = u0 + us_l * span
            pts = np.asarray([self.CU(u) for u in us_g], dtype=float)
            s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            self._LUTs.append((us_l, s))

        # multi：LUTs 逐段一项，path.arcLen 应为所有运动段弧长之和
        #（停顿段为 None 哨兵，跳过）
        self.path.arcLen = float(sum(lut[1][-1] for lut in self._LUTs if lut is not None))

        return self
