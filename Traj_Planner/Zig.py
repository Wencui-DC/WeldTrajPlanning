from Vis import Vis
import numpy as np
from DataClass import *

# ==============================
# Zig 类 (继承 Vis)
# ==============================
class Zig(Vis):
    def __init__(self, wavePara:WavePara, waveNum: float = 1.0):
        super().__init__(wavePara)
        self.computeV()
        self.computeNewT()
        self.computeNewWL()
        self.computeCP()
        self.computeTtList()
        self.initItp()
        self.buildPathSegments(waveNum)
        self.buildArcLengthLUT()



    # 计算关键点 CP
    def computeCP(self):
        self.calcCP0()
        self.calcCP1()
        self.calcCP7()
        self.calcCP6()
        self.calcCP5()
        self.calcCP4()
        self.calcCP2andCP3()


    # --- 子函数 ---
    def calcCP0(self):
        k = self.para.ampLeft / (self.para.ampRight + self.para.ampLeft)
        cp0_x = k * self.newWL * 0.5
        if self.para.tiltAngle > 1e-6:
            cp0_x += np.tan(np.radians(self.para.tiltAngle)) * self.para.ampLeft
        self.CP[0] = np.array([cp0_x, self.para.ampLeft, 0.0])

    def calcCP1(self):
        cp1 = self.CP[0].copy()
        if self.para.dwell_left > 1e-6 and self.para.isMovingWhenDwell:
            cp1[0] += self.v * self.para.dwell_left
        self.CP[1] = cp1

    def calcCP7(self):
        self.CP[7] = np.array([self.para.len, 0.0, 0.0])

    def calcCP6(self):
        cp6 = self.CP[7].copy()
        if abs(self.para.dwell_end) > 1e-6 and self.para.isMovingWhenDwell:
            cp6[0] -= self.v * self.para.dwell_end
        self.CP[6] = cp6

    def calcCP5(self):
        cp6 = self.CP[6].copy()
        k = self.para.ampRight / (self.para.ampLeft + self.para.ampRight)
        cp5_x = cp6[0] - self.newWL * k * 0.5
        if abs(self.para.tiltAngle) > 1e-6:
            cp5_x -= np.tan(np.radians(self.para.tiltAngle)) * self.para.ampRight
        self.CP[5] = np.array([cp5_x, -self.para.ampRight, 0.0])

    def calcCP4(self):
        cp4 = self.CP[5].copy()
        if abs(self.para.dwell_right) > 1e-6 and self.para.isMovingWhenDwell:
            cp4[0] -= self.v * self.para.dwell_right
        self.CP[4] = cp4

    def calcCP2andCP3(self):
        cp4 = self.CP[4].copy()
        cp1 = self.CP[1].copy()
        v1 = cp4 - cp1
        v1Norm = np.linalg.norm(v1)
        if v1Norm < 1e-9:
            v1 = np.zeros(3)
        else:
            v1 /= v1Norm

        k = self.para.ampLeft / (self.para.ampLeft + self.para.ampRight)
        v2 = v1Norm * k * v1
        cp2 = cp1 + v2
        cp3 = cp2.copy()

        if abs(self.para.dwell_mid) > 1e-6 and self.para.isMovingWhenDwell:
            sign = np.sign(cp1[0] - cp4[0])
            v3 = np.array([sign, 0, 0])
            midDis = self.v * self.para.dwell_mid
            v31 = v3 * k * midDis
            v32 = v3 * (k-1) * midDis
            if sign < 0:
                cp2 += v31
                cp3 += v32
            else:
                cp2 += v32
                cp3 += v31
        self.CP[2] = cp2
        self.CP[3] = cp3