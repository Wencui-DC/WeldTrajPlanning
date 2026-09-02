import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import numpy as np
from Vis import Vis
from DataClass import WavePara

class Sine(Vis):
    def __init__(self, wavePara: WavePara, waveNum: float = 1.0):
        super().__init__(wavePara)
        
        self.computeV()
        self.computeNewT()
        self.computeNewWL()
        self.omega = 2 * np.pi / self.newT 
        self.computeCP()
        self.computeTtList() 
        self.initItp()
        self.buildPathSegments(waveNum)
        self.buildArcLengthLUT()

    # ==============================
    # 计算所有关键点 CP
    # ==============================
    def computeCP(self):
        self.calcCP0()
        self.calcCP1()
        self.calcCP2()
        self.calcCP3()
        self.calcCP4()
        self.calcCP5()
        self.calcCP6()
        self.calcCP7()
    # ==============================
    # CP 子函数
    # ==============================
    def calcCP0(self):
        t = self.newT
        self.CP[0] = self.evalSine(t/4)

    def calcCP1(self):
        cp1 = self.CP[0].copy()
        if self.para.dwell_left > 1e-6 and self.para.isMovingWhenDwell:
            cp1[0] += self.v * self.para.dwell_left
        self.CP[1] = cp1

    def calcCP2(self):
        t = self.newT
        cp2 = self.evalSine(t/2)
        if self.para.dwell_left > 1e-6 and self.para.isMovingWhenDwell:
            cp2[0] += self.v * self.para.dwell_left
        self.CP[2] = cp2

    def calcCP3(self):
        cp3 = self.CP[2].copy()
        if self.para.dwell_mid > 1e-6 and self.para.isMovingWhenDwell:
            cp3[0] += self.v * self.para.dwell_mid
        self.CP[3] = cp3

    def calcCP4(self):
        t = self.newT
        cp4 = self.evalSine(t / 4 * 3)
        tempP = self.evalSine(t / 2)
        diff = cp4 - tempP
        xDiff = diff[0]
        cp3 = self.CP[3]
        cp4[0] = cp3[0] + xDiff
        self.CP[4] = cp4

    def calcCP5(self):
        cp5 = self.CP[4].copy()
        if self.para.dwell_right > 1e-6 and self.para.isMovingWhenDwell:
            cp5[0] += self.v * self.para.dwell_right
        self.CP[5] = cp5

    def calcCP6(self):
        t = self.newT 
        cp6 = self.evalSine(t)
        tempP = self.evalSine(t / 4 * 3)
        diff = cp6 - tempP
        xDiff = diff[0]
        cp5 = self.CP[5].copy()
        cp6[0] = cp5[0] + xDiff
        self.CP[6] = cp6

    def calcCP7(self):
        cp7 = self.CP[6].copy()
        if self.para.dwell_end > 1e-6 and self.para.isMovingWhenDwell:
            cp7[0] += self.v * self.para.dwell_end
        self.CP[7] = cp7


    # ==============================
    # 正弦波求值
    # ==============================
    def evalSine(self, t):
        T = self.newT
        t1 = np.mod(t, T)
        if 0 <= t1 < T / 2:
            return self.tiltedSine(self.para.ampLeft, t)
        else:
            return self.tiltedSine(self.para.ampRight, t)

    # ==============================
    # 倾斜正弦计算
    # ==============================
    def tiltedSine(self, amp, t):
        tilt_angle = np.deg2rad(self.para.tiltAngle)
        cos_alpha = np.cos(tilt_angle)

        if abs(cos_alpha) < 1e-6:
            tilt_angle = 0
            amp = -amp

        sign = np.sign(cos_alpha)
        common_part = amp * np.sin(self.omega * t) * sign

        x = np.tan(tilt_angle) * common_part + self.v * t
        y = common_part
        return np.array([x, y, 0.0])