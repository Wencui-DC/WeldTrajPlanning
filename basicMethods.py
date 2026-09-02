import numpy as np
class basicMethods:
    def __init__(self):
        pass

    @staticmethod
    def uniqArr(arr: list[int]):
        _, idx = np.unique(np.round(arr, decimals=8), axis=0, return_index=True)
        arr = arr[np.sort(idx)]
        return arr