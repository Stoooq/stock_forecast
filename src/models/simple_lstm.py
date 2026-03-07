import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_size: int
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)

        last_time_step_out = out[:, -1, :]

        prediction = self.fc(last_time_step_out)

        return prediction
