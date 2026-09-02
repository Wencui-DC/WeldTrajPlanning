from sPlanner import *

if __name__ == "__main__":
    def test(label, arcLen, vMax, aMax, jMax, T):
        vp = sPlanner()
        try:
            vp.planFixedTime(arcLen, vMax, aMax, jMax, T)
            
            print(f"\n {label} ")
            print(vp.summary())
            vp.plot()
        except Exception as e:
            print(f"\n=== {label} FAILED ===\n{e}")

    test("测试1", arcLen=100, vMax=20, aMax=20, jMax=100, T=10)
