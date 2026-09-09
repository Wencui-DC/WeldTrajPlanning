from dataclasses import dataclass, field

@dataclass
class WavePara:
    T = 0.0                 # 摆动周期，单位秒
    len = 0.0               # 波长
    ampLeft = 0.0           
    ampRight = 0.0
    tiltAngle = 0.0         # 偏角，顺时针+，逆时针-
    dwell_left = 0.0
    dwell_mid = 0.0
    dwell_right = 0.0
    dwell_end = 0.0
    isMovingWhenDwell = False
    waveNum = 1.0              # 波型数量（摆动次数）

@dataclass
class WeldData:
    startPose: list = field(default_factory=list)  # 起始位姿 1x6 [x, y, z, rx, ry, rz]
    endPose: list = field(default_factory=list)    # 结束位姿 1x6 [x, y, z, rx, ry, rz]
    vX: float = 0.0                 # x轴进给速度 mm/s
    arcOn: bool = False             # 起弧信号
    vertiWeld: bool = False         # 是否立焊
    current: float = 0              # 焊接电流
    voltage: float = 0              # 焊接电压

    def __post_init__(self):
        # 初始化后自动校验
        self._validate_pose(self.startPose, "startPose")
        self._validate_pose(self.endPose, "endPose")

    def _validate_pose(self, pose: list, name: str):
        # 允许为 None
        if pose is None or len(pose) == 0:
            return
            
        # 必须是 list
        if not isinstance(pose, list):
            raise TypeError(f"{name} 必须是 Python 原生 list 类型")
        
        # 必须长度 = 6
        if len(pose) != 6:
            raise ValueError(f"{name} 必须是 1×6 位姿，当前长度：{len(pose)}")
        
        # 每个值必须是数字
        for val in pose:
            if not isinstance(val, (int, float)):
                raise ValueError(f"{name} 包含无效数值：{val}")


@dataclass
class WeaveData:
    weave_asf: int = 0              # 摆动工艺包编号
    weave_model: int = 0            # 摆动模型
    frequency: float = 0              # 摆动频率
    fwd_angle: int  = 0              # 摆动方向
    pat_angle: int = 0              # 摆动角
    pat_hori: float = 0.0               # 摆幅（后）右
    pat_vert: float = 0.0               # 摆幅（前）左
    dwell_time_left: float = 0.0        # 停留时间前左
    dwell_time_right: float = 0.0       # 停留时间后右
    dwell_time_mid: float = 0.0         # 停留时间中
    dwell_time_end: float = 0.0         # 停留时间末尾
    move_when_dwell: bool = False       # 停留时是否前进


@dataclass
class AstPara:
    enable_horiz: bool = False
    enable_verti: bool = False
    baseRadian: float = 0.0
    horizKp: float = 0.0
    horizKi: float = 0.0
    horizKd: float = 0.0
    vertiKp: float = 0.0
    vertiKi: float = 0.0
    vertiKd: float = 0.0
    dt: float = 0.0
    num_SkippedCycles: int = 0      # 前面几个需要跳过的周期
    num_BaseNeededCycles: int = 0   # 采集基准电流参数所需的周期
    window: int = 0                 # 所需电流的时间窗，单侧


@dataclass
class AstWorkData:
    horizIBase: float = 0.0
    vertiIBase: float = 0.0
    tStart = 0.0
    cycleNum: int = 0
    base_collect_count: int = 0
    collect_horiz = []
    collect_verti = []
    

@dataclass
class iTimeInterval:
    left = (0.0, 0.0)
    mid = (0.0, 0.0)
    right = (0.0, 0.0)
    end = (0.0, 0.0)


@dataclass
class Segment:
    index: int
    start: list = field(default_factory=list)
    end: list = field(default_factory=list)
    start_u: float = 0.0
    end_u: float = 0.0
    duration: float = 0.0
    seg_arc_len: float = 0.0


@dataclass
class Path:
    # Path 包含一个或多个 Segment
    segments: list = field(default_factory=list)
    T: float = 0.0  # 周期总时间
    len: float = 0.0  # 起点到终点的直线距离
    arcLen: float = 0.0 # 总弧长





