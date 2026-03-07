import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout_prob=0.2):
        super().__init__()
        self.dropout_prob = dropout_prob

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = F.dropout(x, p=self.dropout_prob, training=True)

        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = F.dropout(out, p=self.dropout_prob, training=True)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        
        return out