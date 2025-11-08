# import numpy as np
# from pandas import DataFrame
# import torch

# def create_sequences(data, window_size):
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:i+window_size])
#         y.append([data[i+window_size, 3]])
#     return np.array(X), np.array(y)

# def prepare_data(data: DataFrame, target_column: str, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    