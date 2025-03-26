import numpy as np


class HistoryLogger():
    def __init__(self, name):
        self.data = list()
        self.name = name
    
    def add(self, data: np.ndarray | float):
        if isinstance(data, np.ndarray):
            self.data.append(
                [np.min(data), np.mean(data), np.max(data)]
            )
        else:
            self.data.append(float(data))
        
    def print(self):
        d = self.data[-1]
        if isinstance(d, list):
            print(f"{self.name}: min={d[0]:.3f}, mean={d[1]:.3f}, max={d[2]:.3f}")
        else:
            print(f"{self.name}: {d:.3f}")