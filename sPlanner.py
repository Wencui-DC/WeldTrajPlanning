import numpy as np


class sPlanner:
    def __init__(self, dt):
        self.arcLen = 0.0
        self.path = None
        self.veloPlans = []
        self.path_T_cum = np.array([])
        self.path_S_cum = np.array([])
        self.planCache = {}  # 段规划去重缓存：(弧长, 时长) -> sPlanner
        self.vMax = 0.0   # 速度上限
        self.aMax = 0.0   # 加速度上限
        self.jMax = 0.0   # 固定跃度
        self.T = 0.0      # 指定总时间，不可变
        self.vAvg = 0.0   # 指定平均速度，不可变
        self.len = 0.0    # 焊缝总长度

        # 实际使用的 v/a/j
        self.v_used = 0.0
        self.a_used = 0.0
        self.j_used = 0.0

        self.Jerk_seg = np.zeros(7)
        self.T_seg = np.zeros(7)
        self.S_cum = np.zeros(8)
        self.V_cum = np.zeros(8)
        self.A_cum = np.zeros(8)
        self.T_cum = np.zeros(8)

        self._ok = False

        # 插补器状态（interpolate 使用；LUT 由外部构建后传入）
        self.dt = dt                # 时间步长   
        self.LUT_u = None           # 等间距参数网格
        self.LUT_s = None           # 对应累计弧长
        self.segLUTs = None         # 多段模式：逐段局部 LUT（见 WaveBase.buildArcLengthLUT）
        self.uLast = 0.0            # 上一次的路径参数
                                    # （single=全局 u；segments=段局部 u∈[0,1]）
        self.sLast = 0.0            # 当前 TCP 弧长位置
        self.arcStep = 0.0          # 本步弧长增量
        self._segIdx = None         # 多段模式：当前所属段索引
        self._segEndT = None        # 各段结束时刻缓存
        self.CU = None              # 路径求值函数（interpolate 输出位置点用）
        self._itpMode = None        # 'single'=单一路径(无门限) / 'segments'=dwell多段(时间窗门限)
        self._tailFrames = 6.0      # 段末收尾窗口帧数（>0 启用：段结束前
                                    # 窗口内按 剩余弧长/剩余时间 等比预分配，
                                    # 消除段末 clamp 造成的空等/离散相位）

    # ================================================================
    # 主入口：固定总时间 T，速度/加速度在限幅内可调
    # ================================================================
    def planFixedTime(self, arcLen: float, vMax: float, aMax: float, jMax: float, T: float):
        self.arcLen = arcLen
        self.vMax = vMax  # 速度上限
        self.aMax = aMax  # 加速度上限
        self.jMax = jMax  # 固定跃度  
        self.T = T

        if arcLen <= 1e-12:
            self.v_used = 0.0
            self.a_used = 0.0
            self.j_used = 0.0
            self._build_segments()
            self._ok = True
            return self

        self._itpMode = "single"   # 单一路径：插补走弦长累加、无时间窗门限

        # -- 1) 用 aMax 求出可用的 v 范围 --
        v_feas = self._v_feasible_max(aMax)
        v_top = min(vMax, v_feas)
        if v_top <= 0:
            raise ValueError(f"弧长 {arcLen:.3f} 过短，无法完成加减速")

        T_top, _ = self._profile_total_time(v_top, aMax)

        # -- 2) 确定 v_used（a 固定为 aMax，v 二分下调拉长时间）--
        self.a_used = aMax
        self.j_used = jMax
        if T_top <= T + 1e-9:
            # T_top <= T: 全参数下完成太快 → 必须下调 v 来"拉长"时间
            v_lo = arcLen / T * 0.1
            self.v_used = self._bisect_v(v_lo, v_top, aMax, T_target=T)
        else:
            # T_top > T: 以最大可行速度 v_top 运行仍超过指定总时间 → 无解
            raise ValueError(
                f"不可行: 以 vMax/aMax 全参数运行仍超过指定总时间，"
                f"最短需 {T_top:.4f}s > T={T:.4f}s"
            )

        # -- 3) 构建 7 段 --
        self._build_segments()
        self._ok = True
        return self

    
    # ================================================================
    # 主入口：规划已经由 WaveBase 展开的完整路径
    # ================================================================
    def plan(self, path, vMax: float, aMax: float, jMax: float):
        if not hasattr(path, "segments"):
            raise ValueError("path 必须包含 segment 对象")

        self.path = path
        self._segEndT = None   # 规划会改变段时间窗，清段时刻缓存
        self.planCache = {}    # 规划参数可能变化，段规划缓存不跨次复用
        self.vMax = vMax
        self.aMax = aMax
        self.jMax = jMax
        self.vAvg = path.len / path.T if path.T > 0 else 0.0
        self.len = path.len

        if len(path.segments) == 1 or all(segment.arc_length > 1e-12 for segment in path.segments):
            return self._plan_single_path(path)

        self._plan_multiple_path(path, vMax, aMax, jMax)
        self._ok = True
        return self

    def _plan_single_path(self, path):
        """Plan a complete path represented by one continuous segment.

        与多段路径统一：连续路径折叠为一个独立的 sPlanner 段计划
        （内部仍为 7 段解析式），存入 path_vPlans。
        这样 at_time() 只需走 path_vPlans 拼接分支，查询接口完全统一。
        注意不能用 path_vPlans = [self]，否则 at_time() 会无限递归。
        """
        self._itpMode = "single"   # 单一路径：插补走 WaveBase0.getUByArc_Lut 弦长累加口径
        segment_plan = sPlanner(self.dt)
        segment_plan.planFixedTime(
            path.arcLen,
            self.vMax,
            self.aMax,
            self.jMax,
            path.T)

        self.veloPlans = [segment_plan]
        self.path_T_cum = np.array([segment_plan.T])
        self.path_S_cum = np.array([0.0, segment_plan.arcLen])

        # 同步外层字段，保持与直接调用 planFixedTime 等价的外部契约
        self.T = segment_plan.T
        self.arcLen = segment_plan.arcLen
        self.v_used = segment_plan.v_used
        self.a_used = segment_plan.a_used
        self.j_used = segment_plan.j_used
        self._ok = True
        return self

    def _plan_multiple_path(self, path, vMax, aMax, jMax):
        """Plan the complete sequence of segments supplied by WaveBase."""
        self._itpMode = "segments"   # dwell 多段：插补走时间窗门限口径
        self.veloPlans = []
        self.path_T_cum = np.array([])
        self.path_S_cum = np.array([])
        self._plan_segments(path.segments, vMax, aMax, jMax)

        self.path_T_cum = np.cumsum([segment_plan.T for segment_plan in self.veloPlans])
        self.path_S_cum = np.concatenate((
            [0.0],
            np.cumsum([
                segment_plan.arcLen for segment_plan in self.veloPlans
            ])
        ))

        self.T = float(self.path_T_cum[-1])
        self.arcLen = float(self.path_S_cum[-1])
        # 多段路径：外层 summary() 的"实际规划"取各段最大值（各段 v_used 不同）
        self.v_used = max((sp.v_used for sp in self.veloPlans), default=0.0)
        self.a_used = max((sp.a_used for sp in self.veloPlans), default=0.0)
        self.j_used = max((sp.j_used for sp in self.veloPlans), default=0.0)

    def _plan_segments(self, segments, vMax, aMax, jMax):
        """Append plans for the supplied consecutive path segments.

        周期波形展开后会重复出现大量"弧长与时长完全相同"的段，逐段调用
        planFixedTime（内部 80 次二分）纯属重复计算。这里按
        (弧长, 时长) 去重：每种段只规划一次，重复段复用同一个 sPlanner
        实例（缓存见 self.planCache）。

        段规划对象在插补阶段只被 at_time() 只读查询——插补状态
        （uLast/sLast/arcStep、LUT、CU）由外层 planner 持有，段对象不保存
        逐帧状态，因此共享是安全的。veloPlans 仍与 segments 一一对应，
        path_T_cum / path_S_cum 与段索引（_segmentAt 等）逻辑不受影响。
        """
        cache = self.planCache
        for segment in segments:
            arc_length = float(segment.arc_length)
            duration = float(segment.duration)
            # 容差归一化：弧长是数值积分结果，几何等价的段在不同 u 位置积分
            # 出的弧长有 ~1e-9 mm 的浮点差异（与 u 位置无关），不放宽就会各自
            # 重算。量化到 1e-6 mm（纳米，远小于微米级加工精度）/ 1e-9 s。
            key = (round(arc_length, 6), round(duration, 9))
            segment_plan = cache.get(key)
            if segment_plan is None:
                segment_plan = sPlanner(self.dt)
                segment_plan.planFixedTime(
                    arc_length, vMax, aMax, jMax, duration)
                cache[key] = segment_plan
            self.veloPlans.append(segment_plan)


    # ================================================================
    # 内部：解析 (v, a) 对应的分段参数（唯一推导入口）
    # ================================================================
    def _resolve_profile_params(self, v, a):
        """返回 (t_j, t_a, L_acc, a_actual, is_full)"""
        j = self.jMax
        vb = a * a / j

        if v >= vb - 1e-12:
            # 完整 7 段
            t_j = a / j
            t_a = max(0.0, v / a - t_j)
            L_acc = v * v / (2.0 * a) + v * a / (2.0 * j)
            return t_j, t_a, L_acc, a, True
        else:
            # 三角型（T2、T6 退化）
            a_actual = np.sqrt(max(1e-12, v * j))
            t_j = a_actual / j
            L_acc = v * np.sqrt(v / j)
            return t_j, 0.0, L_acc, a_actual, False

    # ================================================================
    # 内部：计算给定 (v, a) 下的总时间与加减速位移
    # ================================================================
    def _profile_total_time(self, v, a):
        t_j, t_a, L_acc, _, _ = self._resolve_profile_params(v, a)

        L_cru = self.arcLen - 2.0 * L_acc
        if L_cru < -1e-9:
            return float('inf'), -1.0               # 不可行: 加减速已超过总弧长
        L_cru = max(0.0, L_cru)
        T_acc = 2.0 * t_j + t_a
        T_cru = L_cru / v if v > 1e-12 else float('inf')
        return 2.0 * T_acc + T_cru, L_acc

    # ================================================================
    # 二分法找 v（给定 a 固定）
    # 前提: 在 [v_lo, v_hi] 上 T(v) 单调递减（慢支）,
    #       且 T(v_lo) >= T_target >= T(v_hi)
    # ================================================================
    def _bisect_v(self, v_lo, v_hi, a_fixed, T_target):
        T_lo, _ = self._profile_total_time(v_lo, a_fixed)
        T_hi, _ = self._profile_total_time(v_hi, a_fixed)

        # 确保边界正确
        if T_lo < T_target - 1e-9:
            v_lo *= 0.5
            T_lo, _ = self._profile_total_time(v_lo, a_fixed)
        if T_hi > T_target + 1e-9:
            v_hi = min(v_hi * 0.8, v_lo + (v_hi - v_lo) * 0.3)
            T_hi, _ = self._profile_total_time(v_hi, a_fixed)

        for _ in range(80):
            v_mid = 0.5 * (v_lo + v_hi)
            T_mid, L_acc = self._profile_total_time(v_mid, a_fixed)
            if L_acc < 0:
                v_hi = v_mid
                continue
            if T_mid > T_target:
                v_lo = v_mid   # 太慢 → 加速
            else:
                v_hi = v_mid   # 太快 → 减速
            if v_hi - v_lo < 1e-12:
                break
        return 0.5 * (v_lo + v_hi)

    # ================================================================
    # 工具函数
    # ================================================================
    def _v_feasible_max(self, a):
        """给定 a，解 2*L_accel = L 得最大可行 v（零巡航）。

        加减速位移公式按 v 相对 vb=a²/j 分两段（见 _resolve_profile_params）：
          - 完整 7 段（v ≥ vb）：L_acc = v²/(2a) + v·a/(2j)
          - 三角型  （v <  vb）：L_acc = v·sqrt(v/j)
        两段分别求零巡航解 2·L_acc = L，取落在各自有效区间的解。
        旧实现只用了 7 段公式，在三角型区域会把可行速度算得极小
        （实测 0.08mm/0.05s 段：0.133mm/s vs 正确 4.31mm/s），
        导致"最短时间 > T"的误报不可行。
        """
        j = self.jMax
        L = self.arcLen
        if L <= 1e-12:
            return 0.0
        vb = a * a / j
        # 完整 7 段零巡航解：v²/a + v·a/j = L
        disc = vb * vb + 4.0 * a * L
        v7 = 0.5 * (-vb + np.sqrt(disc)) if disc >= 0 else 0.0
        # 三角型零巡航解：2·v·sqrt(v/j) = L
        v_tri = (L * np.sqrt(j) / 2.0) ** (2.0 / 3.0)
        cands = []
        if v7 >= vb - 1e-12:
            cands.append(v7)
        if v_tri <= vb + 1e-12:
            cands.append(v_tri)
        if not cands:
            return max(v7, v_tri)
        return max(cands)

    # ================================================================
    # 构建 7 段
    # ================================================================
    def _build_segments(self):
        j = self.j_used
        v = self.v_used
        a = self.a_used
        L = self.arcLen

        if L <= 1e-12:
            self.T_seg = np.array([self.T, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.Jerk_seg.fill(0.0)
            self.T_cum[0] = 0.0
            self.T_cum[1:] = self.T
            self.S_cum.fill(0.0)
            self.V_cum.fill(0.0)
            self.A_cum.fill(0.0)
            self.a_used = 0.0
            self.v_used = 0.0
            return

        t_j, t_a, L_acc, a_used_actual, is_full = self._resolve_profile_params(v, a)

        L_cru = L - 2.0 * L_acc
        t_cru = max(0.0, L_cru / v if v > 1e-12 else 0.0)

        s = [0.0] * 8
        vel = [0.0] * 8
        acc = [0.0] * 8

        if is_full:
            # === 完整 7 段 ===
            # T1
            v1 = 0.5 * j * t_j * t_j
            s[1] = j * t_j**3 / 6.0
            vel[1], acc[1] = v1, a_used_actual
            # T2
            vel[2] = v1 + a_used_actual * t_a
            s[2] = s[1] + v1 * t_a + 0.5 * a_used_actual * t_a * t_a
            acc[2] = a_used_actual
            # T3
            vel[3] = v
            s[3] = s[2] + vel[2] * t_j + 0.5 * a_used_actual * t_j**2 - j * t_j**3 / 6.0
            acc[3] = 0.0
            # T4 巡航
            s[4] = s[3] + v * t_cru
            vel[4], acc[4] = v, 0.0
            # T5
            vel[5] = v - 0.5 * j * t_j * t_j
            s[5] = s[4] + v * t_j - j * t_j**3 / 6.0
            acc[5] = -a_used_actual
            # T6
            vel[6] = vel[5] - a_used_actual * t_a
            s[6] = s[5] + vel[5] * t_a - 0.5 * a_used_actual * t_a * t_a
            acc[6] = -a_used_actual
            # T7
            vel[7] = 0.0
            s[7] = s[6] + vel[6] * t_j - 0.5 * a_used_actual * t_j**2 + j * t_j**3 / 6.0
            acc[7] = 0.0

            self.T_seg = np.array([t_j, t_a, t_j, t_cru, t_j, t_a, t_j])
            self.Jerk_seg = np.array([j, 0.0, -j, 0.0, -j, 0.0, j])

        else:
            # === 三角型（T2=T6=0） ===
            # T1
            v1 = 0.5 * j * t_j * t_j            # = v/2
            s[1] = j * t_j**3 / 6.0
            vel[1], acc[1] = v1, a_used_actual
            # T2 (退化)
            s[2], vel[2], acc[2] = s[1], v1, a_used_actual
            # T3
            vel[3] = v
            s[3] = s[2] + v1 * t_j + 0.5 * a_used_actual * t_j**2 - j * t_j**3 / 6.0
            acc[3] = 0.0
            # T4 巡航
            s[4] = s[3] + v * t_cru
            vel[4], acc[4] = v, 0.0
            # T5
            vel[5] = v - 0.5 * j * t_j * t_j    # = v/2
            s[5] = s[4] + v * t_j - j * t_j**3 / 6.0
            acc[5] = -a_used_actual
            # T6 (退化)
            s[6], vel[6], acc[6] = s[5], vel[5], -a_used_actual
            # T7
            vel[7] = 0.0
            s[7] = s[6] + vel[6] * t_j - 0.5 * a_used_actual * t_j**2 + j * t_j**3 / 6.0
            acc[7] = 0.0

            self.T_seg = np.array([t_j, 0.0, t_j, t_cru, t_j, 0.0, t_j])
            self.Jerk_seg = np.array([j, 0.0, -j, 0.0, -j, 0.0, j])

        # 累计量
        self.T_cum[0] = 0.0
        for k in range(7):
            self.T_cum[k + 1] = self.T_cum[k] + self.T_seg[k]
        self.S_cum = np.array(s)
        self.V_cum = np.array(vel)
        self.A_cum = np.array(acc)

        # 保存实际使用的 a（三角型可能小于 aMax）
        self.a_used = a_used_actual

    # ================================================================
    # 查询
    # ================================================================
    def at_time(self, t: float):
        """返回 (s, v, a)"""
        if not self._ok:
            raise RuntimeError("请先调用 plan()")

        if self.veloPlans:
            t = max(0.0, min(t, self.T))
            index = int(np.searchsorted(self.path_T_cum, t, side="right"))
            index = min(index, len(self.veloPlans) - 1)
            start_t = 0.0 if index == 0 else self.path_T_cum[index - 1]
            local_s, velocity, acceleration = self.veloPlans[index].at_time(
                t - start_t)
            return self.path_S_cum[index] + local_s, velocity, acceleration

        t = max(0.0, min(t, self.T))
        k = self._seg_index(t)
        dt = t - self.T_cum[k]

        jk = self.Jerk_seg[k]
        a0 = self.A_cum[k]
        v0 = self.V_cum[k]
        s0 = self.S_cum[k]

        a = a0 + jk * dt
        v = v0 + a0 * dt + 0.5 * jk * dt * dt
        s = s0 + v0 * dt + 0.5 * a0 * dt * dt + jk * dt**3 / 6.0
        return s, v, a

    def v_at(self, t):
        return self.at_time(t)[1]

    def s_at(self, t):
        return self.at_time(t)[0]

    def _seg_index(self, t):
        lo, hi = 0, 6
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.T_cum[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ================================================================
    # 插补：沿已规划路径推进（配合外部构建的弧长 LUT）
    # 由 WaveBase 的 getUByArc_Lut 及相关函数迁移而来：sPlanner 持有
    # 速度规划 v_at(t)，只需外部传入 LUT(u↔s) 即可独立完成插补。
    # ================================================================
    def interpolate(self, wave, t):
        """沿路径推进一个插补步，返回位置点 posi = C(u)。

        :param wave: 波形对象，须提供 LUT_u/LUT_s 弧长查找表与 CU 求值函数
                     （如 Sine 实例）；多段（dwell 停顿）路径须提供
                     segLUTs（逐段局部 LUT）。
        :param t:  当前绝对时间（秒），用于段时间窗门限判定。
        :return: 位置点 posi（3 元 ndarray）；同时更新 self.uLast /
                 self.sLast / self.arcStep。速度取自已完成的规划 v_at(t)。
        """
        if not self._ok:
            raise RuntimeError("请先调用 plan()")

        self.CU = getattr(wave, "CU", None)
        if self.CU is None:
            raise ValueError(
                "缺少路径求值函数 CU：无法输出位置点，请传入带 CU 的波形对象")

        if self._itpMode == "segments":
            # —— 多段：按 t 定位段 → 该段规划 → 段内 u∈[0,1] 单段插补 ——
            return self._interpSegments(wave, t)
        else:
            # —— 单一路径（single）：原逻辑原样，不做任何改动 ——
            u_arr, s_arr = self._unpackLUT(wave)
            self.LUT_u = u_arr
            self.LUT_s = s_arr
            u = self.getUByArc_Lut(t)
            return self._CU_ext(u)

    def _interpSegments(self, wave, t):
        """多段路径插补：每段各自当作一条"单路段"来走。

        1) 由 t 定位所属段 idx（沿用 _segmentEndTimes 的段时间窗）；
        2) 速度取该段自己的规划 veloPlans[idx].at_time(t_local)
           （t_local = t - 段起始时刻，段内时间轴从 0 开始）；
        3) 调用单段插补接口 _stepSegment：弦长推进 + 弦长迭代，
           s↔u 映射用该段的局部 LUT，u 的取值恒为段局部 [0, 1]；
        4) 跨段（idx 变化）时 uLast 重置为 0 —— 每段都从自己的 0 开始，
           不沿用上一段的参数；
        5) 输出位置：段局部 u → 全局 u = start_u + u_local·(end_u-start_u)。
        """
        segments = self.path.segments if self.path is not None else []
        if not segments:
            raise ValueError("多段路径为空：path.segments 为空")

        segLUTs = getattr(wave, "segLUTs", None)
        if segLUTs is None:
            raise ValueError(
                "多段路径插补需要 wave.segLUTs（逐段局部 LUT），"
                "请由 WaveBase.buildArcLengthLUT 构建")
        self.segLUTs = segLUTs

        t_now = float(t)
        ends = self._segmentEndTimes()
        idx = int(np.searchsorted(ends, t_now, side="right"))
        idx = min(idx, len(segments) - 1)
        seg = segments[idx]
        seg_start_t = 0.0 if idx == 0 else float(ends[idx - 1])

        # 跨段：段内参数回到 0（uLast 在多段模式下是段局部 u∈[0,1]）
        if self._segIdx != idx:
            self._segIdx = idx
            self.uLast = 0.0

        segLUT = segLUTs[idx] if idx < len(segLUTs) else None
        # 停顿段（弧长=0 / 无局部 LUT）：原地停在段起点，不推进
        if float(seg.arc_length) <= 1e-12 or segLUT is None:
            self.arcStep = 0.0
            self.sLast = float(self.path_S_cum[idx])
            return np.asarray(self.CU(float(seg.start_u)))

        if self.dt <= 1e-12:
            u_local = float(self.uLast)
        else:
            u_local = self._stepSegment(
                seg, segLUT, self.veloPlans[idx],
                t_now - seg_start_t, self.dt, float(self.uLast))

        self.uLast = u_local
        s_local = float(np.interp(u_local, segLUT[0], segLUT[1]))
        s_global = float(self.path_S_cum[idx]) + s_local
        self.arcStep = max(0.0, s_global - self.sLast)
        self.sLast = s_global

        return self._segCU(seg, u_local)

    def _stepSegment(self, seg, segLUT, plan, t_local, dt, u_start):
        """单段插补接口（段局部 u∈[0,1]，与 _stepSinglePath 同一推进口径）。

        vt = 该段规划在段内时刻 t_local 的速度，ds = vt·dt 累加，
        再做弦长迭代（20 次 / tol=1e-6）修正到弦速 = vt。
        与 _stepSinglePath 的差别只在 s↔u 映射：这里用段局部 LUT
        （u 恒在 [0,1]，不过冲、不跨段），速度查询走段自身的规划。
        """
        vt = float(plan.v_at(t_local))
        if vt <= 0.0:
            return u_start

        us_l, ss = segLUT
        s_current = float(np.interp(u_start, us_l, ss))
        ds = vt * dt
        u_next = float(np.interp(s_current + ds, ss, us_l))
        for _ in range(20):
            chord_len = np.linalg.norm(
                self._segCU(seg, u_next) - self._segCU(seg, u_start))
            v_chord = chord_len / dt
            if abs(v_chord - vt) <= 1e-6:
                break
            ds *= vt / max(v_chord, 1e-12)    # 比例修正：弦速偏小则加大 ds
            u_next = float(np.interp(s_current + ds, ss, us_l))

        return float(min(max(u_next, u_start), 1.0))

    def _segCU(self, seg, u_local):
        """段局部 u∈[0,1] → 全局 u → 位置点。"""
        u_global = (float(seg.start_u)
                    + float(u_local) * (float(seg.end_u) - float(seg.start_u)))
        return np.asarray(self.CU(u_global))

    def _CU_ext(self, u):
        """路径求值：u 未过终点用波形 CU；u 越过终点沿终点切向延长。

        过冲段不采用波形周期延拓——整数 numWave + dwell 时终点落在周期
        边界，dwell_left 爬升段使波形在该处几何不连续，延拓会造成位置跳变
        （末段弦速瞬间放大数百%）。改沿终点切线平滑外推，与
        _s2u_ext/_u2absS 的末端斜率外推自洽（|ΔC|≈Δs），弦长迭代可收敛。
        """
        u = float(u)
        u_end = float(self.LUT_u[-1])
        if u <= u_end:
            return np.asarray(self.CU(u))
        du = float(self.LUT_u[-1] - self.LUT_u[-2])
        p_end = np.asarray(self.CU(u_end))
        p_prev = np.asarray(self.CU(float(self.LUT_u[-2])))
        tangent = (p_end - p_prev) / max(du, 1e-16)   # dC/du ≈ 终点切向量
        return p_end + (u - u_end) * tangent

    @staticmethod
    def _unpackLUT(wave):
        """归一化波形对象为 (LUT_u, LUT_s) 两个 ndarray。"""
        return (np.asarray(wave.LUT_u, dtype=float),
                np.asarray(wave.LUT_s, dtype=float))

    def resetInterp(self):
        """复位插补状态：路径参数与弧长进度。"""
        self.uLast = 0.0
        self.sLast = 0.0
        self.arcStep = 0.0
        self._segIdx = None   # 多段模式：当前段索引（跨段重置 uLast 用）

    def _segmentEndTimes(self):
        """各段结束时刻的累计值（缓存）。"""
        if self._segEndT is None:
            if self.path is None or not self.path.segments:
                self._segEndT = np.array([])
            else:
                self._segEndT = np.cumsum(
                    [float(seg.duration) for seg in self.path.segments])
        return self._segEndT

    def _segmentAt(self, t):
        """时刻 t 所属的路径段。"""
        segments = self.path.segments if self.path is not None else []
        if not segments:
            return None
        index = int(np.searchsorted(self._segmentEndTimes(), t, side="right"))
        if index >= len(segments):
            index = len(segments) - 1
        return segments[index]

    def _segEndTimeAt(self, t):
        """时刻 t 所属路径段的结束时刻（段末收尾窗口用）。"""
        ends = self._segmentEndTimes()
        if ends is None or len(ends) == 0:
            return None
        i = int(np.searchsorted(ends, t, side="right"))
        i = min(i, len(ends) - 1)
        return float(ends[i])

    def _uGate(self, t):
        """时刻 t 允许到达的最远 u：由"本段的时间窗"给出。

        规划把总时间切给各段：移动段走几何，停顿段原地等待。机器人不得越过
        **当前时间窗所属段**的末端——否则就等于提前进入下一段的几何，把其间
        的停顿整个跳过。这正是单纯按弦长推进在停顿路径上失效的原因。
        """
        u_max = float(self.LUT_u[-1])
        segment = self._segmentAt(t)
        if segment is None:
            return u_max
        return min(float(segment.end_u), u_max)

    def getUByArc_Lut(self, t):
        """沿统一全局 U 轴推进一个插补步，按路径形态分流：

        - 单一路径（isMovingWhenDwell=True 展开的单段连续路径）：
          WaveBase0.getUByArc_Lut(vt, dt) 口径——vt = sVelo.at_time(t)（规划
          当前速度），ds = vt*dt 纯累加 + 弦长迭代（20 次 / tol=1e-6），
          无时间窗门限。

        - 多段路径（isMovingWhenDwell=False 的 dwell 停顿路径）：
          时间窗门限 _uGate 生效，不得越过当前时间窗所属段的末端
          （dwell 窗口内 vt=0，u 自然停留等待）。
        """
        dt = self.dt
        if dt <= 1e-12:
            return float(self.uLast)

        t_now = float(t)
        u_start = float(self.uLast)
        u_max = float(self.LUT_u[-1])

        # 允许终点过冲的情形：
        #   - single 模式（单段连续路径，老算法行为）；
        #   - 多段模式的最后一个移动段（arc>0 且 end_u 即路径终点，其后无
        #     dwell 停顿，过冲不会跳过任何等待）。中间移动段必须钳在段末，
        #     否则 TCP 提前进入下一段几何、把 dwell 停顿整个跳过；而停顿段
        #     弧长=0（u 区间是虚拟区间，推进 u 不产生位移），过冲必然失控。
        if self._itpMode == "single":
            allow_overshoot = True
        else:
            seg = self._segmentAt(t_now + dt)
            allow_overshoot = (
                seg is not None
                and float(getattr(seg, "arc_length", 0.0)) > 1e-9
                and float(seg.end_u) >= u_max - 1e-9)

        if u_start >= u_max and not allow_overshoot:
            return u_start

        if allow_overshoot:
            return self._stepSinglePath(t_now, dt, u_start)

        u_limit = self._uGate(t_now + dt)
        if u_limit <= u_start:
            return u_start

        # —— 段起点对齐 + 段比例 α + 弦长/路段平均 vt 比对 ——
        # 目标折线弧长 = 本段折线起点 + (s_at(t+dt)-s_at(段起点))·α：
        # 解析弧长增量按段长比例 α = LUT 折线段长/解析段长(arc_length) 折进
        # 本段折线标尺（折线弦长比精确弧长积分短 ~0.01mm/段）。段末 t=段规划
        # 末时折线目标恰好到达几何端 end_u → 几何与时间同步耗尽。
        # 迭代目标取路段平均 (v_at(t)+v_at(t+dt))/2 而非端点瞬时：减速段端点
        # 瞬时 > 路段平均，逐帧拉大 ds 会把 u 提前耗尽段末几何 → 段界空等帧
        # dev 69/88/100%；平均口径与解析差分推进一致，段界相位干净。
        # （single 分支巡航为主、瞬时与平均等价，保持瞬时不变。）
        seg_now = self._segmentAt(t_now)
        if (seg_now is None
                or float(getattr(seg_now, "arc_length", 0.0)) <= 1e-9):
            return u_start
        u_geom_end = float(min(seg_now.end_u, u_max))
        if u_geom_end <= u_start + 1e-12:
            return u_start

        v_now = float(self.v_at(t_now))
        v_tgt = float(self.v_at(t_now + dt))
        vt_avg = 0.5 * (v_now + v_tgt)
        if vt_avg <= 0.0:
            return u_start

        ends = self._segmentEndTimes()
        idx = int(np.searchsorted(ends, t_now, side="right"))
        idx = min(idx, len(ends) - 1)
        seg_start_t = 0.0 if idx == 0 else float(ends[idx - 1])

        s_lut_seg0 = float(self._u2s_lut(float(seg_now.start_u)))
        lut_len = float(self._u2s_lut(float(seg_now.end_u))) - s_lut_seg0
        plan_len = max(float(seg_now.arc_length), lut_len, 1e-12)
        alpha = lut_len / plan_len
        s_lut_tgt = s_lut_seg0 + (float(self.s_at(t_now + dt))
                                  - float(self.s_at(seg_start_t))) * alpha

        s_current = self._u2absS(u_start)
        u_next = float(min(max(self._s2u_lut(s_lut_tgt), u_start), u_geom_end))
        for _ in range(20):
            if u_next >= u_geom_end - 1e-12:
                break
            chord_len = np.linalg.norm(self.CU(u_next) - self.CU(u_start))
            v_chord = chord_len / dt
            if abs(v_chord - vt_avg) <= 1e-6:
                break
            s_lut_tgt += max(s_lut_tgt - s_current, 1e-12) * (
                vt_avg / max(v_chord, 1e-12) - 1.0)
            u_next = float(min(max(self._s2u_lut(s_lut_tgt), u_start),
                               u_geom_end))

        self.sLast = self._u2absS(u_next)
        self.arcStep = max(0.0, self.sLast - s_current)
        self.uLast = u_next
        return u_next

    def getUBySeg(self, t):
        """多段（dwell 停顿）路径的逐段推进（旧口径，interpolate 已改用
        _interpSegments/_stepSegment：走段自身规划 + 单段插补接口）。

        每个移动段视为一个独立的"单路段"：段内弧长标尺取自该段的局部
        LUT（u_local∈[0,1] 的弦长折线，端长拉伸标定到段解析弧长
        arc_length）。每帧推进量 = 规划弧长差分 ds = s_at(t+dt)-s_at(t)
        —— 即该路段平均弧长（"速度与规划一致"的正确路段平均口径），
        反查得段内 u_local 再映射回全局 u。

        停顿段（arc=0，几何不动）：不生成 LUT，原地停在段起点
        （= 上一移动段的几何端，即停顿平台点），速度 v→0 自然停留。

        端长标定：局部 LUT 折线端比解析弧长短 δ≈6~8.5um/段（1000 点弦
        折线的弓差）。若直接用折线端判段末，几何端会提前 δ 到达 → 段末
        空等帧（dev 88/100%）。故按 ratio = arc_length/折线端把推进弧长
        统一到解析标尺：段末解析差分 ds→0 与 v→0 同刻，几何端即停，
        无提前、无空等、无段界相位帧。
        """
        dt = self.dt
        t_now = float(t)
        if dt <= 1e-12:
            return float(self.uLast)
        ends = self._segmentEndTimes()
        idx = int(np.searchsorted(ends, t_now, side="right"))
        idx = min(idx, len(self.path.segments) - 1)
        seg = self.path.segments[idx]
        segLUT = self.segLUTs[idx] if self.segLUTs is not None else None

        u_lo = float(seg.start_u)
        u_hi = float(seg.end_u)

        # —— 停顿段（arc=0）：不推进，原地停 ——
        if float(seg.arc_length) <= 1e-12 or segLUT is None:
            return float(min(max(self.uLast, u_lo), u_hi))

        # —— 移动段：段内弧长推进 ——
        u_cl = float(min(max(self.uLast, u_lo), u_hi))
        if u_cl >= u_hi - 1e-12:
            return u_hi

        ds = float(self.s_at(t_now + dt) - self.s_at(t_now))
        if ds <= 0.0:
            return u_cl

        us_l, ss_raw = segLUT
        arc = float(seg.arc_length)
        ratio = arc / float(ss_raw[-1])     # 端长标定：拉伸折线到解析弧长
        u_span = u_hi - u_lo
        u_loc0 = (u_cl - u_lo) / u_span     # 段内当前参数 ∈[0,1)
        s0_raw = float(np.interp(u_loc0, us_l, ss_raw))
        s1_raw = s0_raw + ds / ratio
        if s1_raw >= float(ss_raw[-1]) - 1e-12:
            u_loc = 1.0                     # 段末：解析弧长耗尽 = 几何端
        else:
            u_loc = float(np.interp(s1_raw, ss_raw, us_l))
        u_loc = max(u_loc0, u_loc)
        u_next = float(min(max(u_lo + u_loc * u_span, u_cl), u_hi))

        self.sLast = float(self.s_at(t_now + dt))
        self.arcStep = ds
        self.uLast = u_next
        return u_next

    def _stepSinglePath(self, t_now, dt, u_start):
        """WaveBase0.getUByArc_Lut(vt, dt) 口径：单一路径无门限弦长推进。

        vt = sVelo.at_time(t)（规划当前速度），ds = vt*dt 纯累加，
        弦长迭代 20 次 / tol=1e-6 修正（与 WaveBase0 完全一致）。
        全局 LUT 坐标系，无需跨周期分解。
        """
        vt = float(self.v_at(t_now))          # 目标 TCP 速度 = 规划当前速度
        if vt <= 0.0:
            return u_start

        s_current = self._u2absS(u_start)
        ds = vt * dt
        u_next = float(self._s2u_ext(s_current + ds))
        for _ in range(20):
            # 不做终点钳制跳出：s 越过 LUT 终点后按末端斜率线性外推 u，
            # 弦长迭代继续收敛到 vt（老算法过冲行为——减速段末尾 u 略超
            # 终点沿切向延长继续走；不会提前停车）。过冲段位置用 _CU_ext
            # 切向延长而非波形周期延拓，避免 dwell 周期边界处位置跳变。
            chord_len = np.linalg.norm(self._CU_ext(u_next) - self._CU_ext(u_start))
            v_chord = chord_len / dt
            if abs(v_chord - vt) <= 1e-6:
                break
            ds *= vt / max(v_chord, 1e-12)    # 比例修正：弦速偏小则加大 ds
            u_next = float(self._s2u_ext(s_current + ds))
        u_next = float(max(u_next, u_start))

        self.sLast = self._u2absS(u_next)
        self.arcStep = max(0.0, self.sLast - s_current)
        self.uLast = u_next
        return u_next

    def _s2u_lut(self, s_local):
        """给定全局弧长 s_local (0 ~ LUT_s[-1])，反查全局参数 u。"""
        if self.LUT_s is None or len(self.LUT_s) < 2:
            return 0.0
        s_local = max(0.0, min(float(s_local), float(self.LUT_s[-1])))
        lo, hi = 0, len(self.LUT_s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.LUT_s[mid] <= s_local:
                lo = mid
            else:
                hi = mid
        frac = (s_local - self.LUT_s[lo]) / (self.LUT_s[hi] - self.LUT_s[lo] + 1e-16)
        frac = max(0.0, min(frac, 1.0))
        return float(self.LUT_u[lo] + frac * (self.LUT_u[hi] - self.LUT_u[lo]))

    def _s2u_ext(self, s):
        """全局弧长→参数 u；s 越过终点 LUT_s[-1] 时按末端斜率线性外推。

        外推让 u 略超 LUT_u[-1]，配合波形对象的周期延拓求值 CU(u) 实现
        老算法的终点过冲（single 模式专用；multi 模式不调用本方法）。
        """
        s_end = float(self.LUT_s[-1])
        if float(s) <= s_end:
            return self._s2u_lut(s)
        du = float(self.LUT_u[-1] - self.LUT_u[-2])
        ds = float(self.LUT_s[-1] - self.LUT_s[-2])
        k = du / max(ds, 1e-16)
        return float(self.LUT_u[-1] + (float(s) - s_end) * k)

    def _u2absS(self, u):
        """Return the arc length at global path parameter u ∈ [0, waveNum].

        u 越过终点 LUT_u[-1] 时按末端斜率线性外推（single 模式过冲时
        sLast 与弧长进度保持一致）。
        """
        u_end = float(self.LUT_u[-1])
        if float(u) <= u_end:
            return self._u2s_lut(u)
        du = float(self.LUT_u[-1] - self.LUT_u[-2])
        ds = float(self.LUT_s[-1] - self.LUT_s[-2])
        k = ds / max(du, 1e-16)
        return float(self.LUT_s[-1] + (float(u) - u_end) * k)

    def _u2s_lut(self, u):
        """LUT 二分 + 线性插值查询 s(u)，u ∈ [0, waveNum]"""
        if self.LUT_u is None or len(self.LUT_u) < 2:
            return 0.0
        u = max(float(self.LUT_u[0]), min(float(u), float(self.LUT_u[-1])))
        lo, hi = 0, len(self.LUT_u) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.LUT_u[mid] <= u:
                lo = mid
            else:
                hi = mid
        frac = (u - self.LUT_u[lo]) / (self.LUT_u[hi] - self.LUT_u[lo] + 1e-16)
        frac = max(0.0, min(frac, 1.0))
        return float(self.LUT_s[lo] + frac * (self.LUT_s[hi] - self.LUT_s[lo]))

    # ================================================================
    # 画图
    # ================================================================
    def plot(self, total_pts=500):
        """绘制 s(t), v(t), a(t), j(t) 四子图，标注 7 段边界
        自动在 jerk 非零段加密采样，避免极短加减速段被遗漏"""
        if not self._ok:
            raise RuntimeError("尚未规划或规划失败，导致无法绘图！")
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            raise ImportError("需要 matplotlib，请 pip install matplotlib")

        # 强制重建字体缓存 + 设置中文字体（Windows 常见坑）
        try:
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # ---- 分段自适应采样 ----
        # jerk 非零段 (T1/T3/T5/T7) 每段至少 dense_each 个点
        # 剩余点数按比例分给 T2/T4/T6
        dense_each = max(20, total_pts // 8)   # 每段非零 jerk 至少 20 点
        jerk_nonzero_idx = [i for i in range(7) if abs(self.Jerk_seg[i]) > 1e-12]
        jerk_zero_idx = [i for i in range(7) if abs(self.Jerk_seg[i]) <= 1e-12 and self.T_seg[i] > 1e-12]

        # 非零段：每段 dense_each 点
        n_dense = len(jerk_nonzero_idx) * dense_each

        # 零 jerk 段（巡航为主）：按时长比例分配剩余点数
        T_zero_total = sum(self.T_seg[i] for i in jerk_zero_idx)
        pts_per_zero = {}
        if total_pts > n_dense and T_zero_total > 0:
            remaining = total_pts - n_dense
            for idx in jerk_zero_idx:
                pts_per_zero[idx] = max(1, int(round(remaining * self.T_seg[idx] / T_zero_total)))

        # 组装时间数组
        t_parts = []
        for k in range(7):
            if k in jerk_nonzero_idx:
                n = dense_each
            else:
                n = pts_per_zero.get(k, 0)
            if n > 0 and self.T_seg[k] > 1e-12:
                t_parts.append(np.linspace(self.T_cum[k], self.T_cum[k + 1], n, endpoint=(k == 6)))
        t = np.concatenate(t_parts)
        n_pts = len(t)

        s_vals = np.empty(n_pts)
        v_vals = np.empty(n_pts)
        a_vals = np.empty(n_pts)
        j_vals = np.empty(n_pts)

        for i, ti in enumerate(t):
            s_vals[i], v_vals[i], a_vals[i] = self.at_time(ti)
            k = self._seg_index(ti)
            j_vals[i] = self.Jerk_seg[k]

        seg_bounds = self.T_cum[1:7]  # T1末, T2末, ..., T6末

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('S 型速度规划', fontsize=14, fontweight='bold')

        # --- s(t) ---
        lw = 2
        ax = axes[0]
        ax.plot(t, s_vals, 'b-', linewidth=lw)
        ax.set_ylabel('位移 S [mm]')
        ax.grid(True, alpha=0.3)

        # --- v(t) ---
        ax = axes[1]
        ax.plot(t, v_vals, 'g-', linewidth=lw)
        ax.set_ylabel('速度 V [mm/s]')
        ax.grid(True, alpha=0.3)

        # --- a(t) ---
        ax = axes[2]
        ax.plot(t, a_vals, 'r-', linewidth=lw)
        ax.set_ylabel('加速度 A [mm/s^2]')
        ax.grid(True, alpha=0.3)

        # --- j(t) ---
        ax = axes[3]
        ax.step(t, j_vals, 'm-', linewidth=lw, where='post')
        ax.set_ylabel('跃度 J [mm/s^3]')
        ax.set_xlabel('时间 t [s]')
        ax.grid(True, alpha=0.3)

        # 在每张子图上画段边界
        for ax in axes:
            for bd in seg_bounds:
                ax.axvline(bd, color='gray', linestyle='--', linewidth=1, alpha=1)

        plt.tight_layout()
        plt.show()
        return fig, axes

    # ================================================================
    # 信息
    # ================================================================
    def summary(self):
        if not self._ok:
            return "未规划"
        lines = [
            "=" * 60,
            f"  S型速度规划",
            f"  弧长 = {self.arcLen:.2f} mm，焊缝长度 = {self.len:.2f} mm，焊接速度 = {self.vAvg:.2f} mm/s， 焊接时间 = {self.T:.2f} s", 
            f"  输入上限: vMax = {self.vMax:.2f} mm/s  aMax = {self.aMax:.2f} mm/s^2  jMax = {self.jMax:.2f} mm/s^3",
            f"  实际规划: vTCP = {self.v_used:.2f} a = {self.a_used:.2f} j = {self.j_used:.2f}",
        ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def printVeloPlans(self):
        """打印各路径段的速度规划及去重复用情况。

        段规划只取决于（弧长, 时长），与段所在的 u 位置无关，
        因此几何/时长相同的段共用同一个 sPlanner 实例。
        """
        if not self._ok:
            print("尚未规划。")
            return
        if not self.veloPlans:
            print("无段速度规划（当前为 planFixedTime 直接规划的单段）。")
            return

        plans = self.veloPlans
        segments = self.path.segments if self.path is not None else []

        # 同一实例 = 同一份规划（去重复用）
        plan_id = {}
        groups = {}
        unique = []
        for i, sp in enumerate(plans):
            pid = plan_id.get(id(sp))
            if pid is None:
                pid = len(unique) + 1
                plan_id[id(sp)] = pid
                unique.append((pid, sp))
                groups[pid] = []
            groups[pid].append(i + 1)

        print("\n" + "=" * 78)
        print(f"速度规划：{len(plans)} 段 / {len(groups)} 个唯一规划"
              f"（复用 {len(plans) - len(groups)} 次）")
        print("-" * 78)
        print(f"{'段':<4}| {'u范围':<18}| {'时长s':>7} | {'弧长mm':>9} | "
              f"{'规划#':<6}| {'v':>8} | {'a':>9} | {'j':>10}")
        print("-" * 78)
        for i, sp in enumerate(plans):
            seg = segments[i] if i < len(segments) else None
            if seg is not None:
                u_str = f"[{seg.start_u:.3f}, {seg.end_u:.3f}]"
                duration = float(seg.duration)
            else:
                u_str = "-"
                duration = float(sp.T)
            print(
                f"{i + 1:<4}| {u_str:<18}| {duration:>7.3f} | "
                f"{float(sp.arcLen):>9.3f} | {plan_id[id(sp)]:<6}| "
                f"{sp.v_used:>8.2f} | {sp.a_used:>9.2f} | {sp.j_used:>10.2f}")
        print("-" * 78)

        print("唯一规划明细（弧长/时长相同即复用，与 u 位置无关）：")
        for pid, sp in unique:
            idxs = groups[pid]
            idx_str = ",".join(str(x) for x in idxs[:8])
            if len(idxs) > 8:
                idx_str += f",…(+{len(idxs) - 8})"
            print(
                f"  #{pid}: 弧长={float(sp.arcLen):.4f} mm, 时长={float(sp.T):.4f} s, "
                f"v={sp.v_used:.3f}, a={sp.a_used:.3f}, j={sp.j_used:.3f}"
                f"  → 段 {idx_str}")

        print("-" * 78)
        print(f"合计：焊接时间 = {self.T:.3f} s，路径弧长 = {self.arcLen:.3f} mm")
        print("=" * 78 + "\n")





