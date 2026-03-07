import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int, target_col: str):
        X_numpy = data.loc[:, data.columns != target_col].to_numpy(dtype=np.float32)
        y_numpy = data[target_col].to_numpy(dtype=np.float32)

        self.X = torch.from_numpy(X_numpy)
        self.y = torch.from_numpy(y_numpy)
        self.seq_len = sequence_length

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]

        return x_window, y_target
