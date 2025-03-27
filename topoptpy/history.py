from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class HistoryLogger():
    def __init__(
        self,
        name: str,
        constants: Optional[list[float]]=None,
        constant_names: Optional[list[str]]=None,
    ):
        self.data = list()
        self.name = name
        self.constants = constants
        self.constant_names = constant_names
    
    def exists(self):
        ret = True if len(self.data) > 0 else False
        return ret
    
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


class HistoriesLogger():
    def __init__(
        self, dst_path: str
    ):
        self.dst_path = dst_path
        self.histories = dict()
    
    def feed_data(self, name: str, data: np.ndarray | float):
        self.histories[name].add(data)
    
    def add(
        self,
        name: str,
        constants: Optional[list[float]] = None,
        constant_names: Optional[list[str]] = None,
    ):
        hist = HistoryLogger(
            name,
            constants=constants,
            constant_names=constant_names
        )
        self.histories[name] = hist
    
    def print(self):
        for k in self.histories.keys():
            if self.histories[k].exists():
                self.histories[k].print()
    
        
    def export_progress(self, fname: Optional[str] = None):
        if fname is None:
            fname = "progress.jpg"
        plt.clf()
        rows, cols = 2, 4
        fig, ax = plt.subplots(rows, cols, figsize=(16, 8))

        for i, k in enumerate(self.histories.keys()):
            h = self.histories[k]
            if h.exists():
                p = i // cols
                q = i % cols
                d = np.array(self.histories[k].data)
                
                if d.ndim > 1:
                    ax[p, q].plot(d[:, 0], marker='o', linestyle='-', label="min")
                    ax[p, q].plot(d[:, 1], marker='o', linestyle='-', label="mean")
                    ax[p, q].plot(d[:, 2], marker='o', linestyle='-', label="max")
                    ax[p, q].legend()
                else:
                    ax[p, q].plot(d, marker='o', linestyle='-')

                ax[p, q].set_xlabel("Iteration")
                ax[p, q].set_ylabel(h.name)
                ax[p, q].set_title(f"{h.name} Progress")
                ax[p, q].grid(True)

        fig.tight_layout()
        fig.savefig(f"{self.dst_path}/{fname}")
        plt.close("all")