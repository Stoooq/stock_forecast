import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.config import Config
from data.data_module import DataModule
from models.model import MyModel
from training.checkpointer import Checkpointer
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from viz.visualizer import Visualizer

cfg = Config("/Users/miloszglowacki/Desktop/code/python/stock_forecast/config/config.yaml")
device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

data_module = DataModule(cfg.data)
train_loader, val_loader, test_loader, scaler, test_dates = data_module.get_loaders()

for x, y in train_loader:
    print(x.shape)

model = MyModel(input_size=55, hidden_size=128, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

checkpointer = Checkpointer(cfg.training)

total_sum = 0
total_count = 0

for x, y in train_loader:
    total_sum += y.sum().item()
    total_count += y.numel()

global_mean = total_sum / total_count

print(f"Sum: {total_sum}, num: {total_count}, mean: {global_mean}")

trainer = Trainer(model, loss_fn, optimizer, cfg.training, checkpointer)
history = trainer.train(train_loader, val_loader)

evaluator = Evaluator(model, cfg.training)
true, pred, res = evaluator.evaluate(test_loader)

print(y[:50], pred[:50])
print("true mean, std:", true.mean(), true.std())
print("pred mean, std:", pred.mean(), pred.std())

mse = mean_squared_error(true, pred)
mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mse)
print("MSE, RMSE, MAE:", mse, rmse, mae)

corr = np.corrcoef(true, pred)[0,1]
print("Pearson corr:", corr)

mean_pred = np.full_like(true, true.mean())
mse_mean = mean_squared_error(true, mean_pred)
print("MSE mean baseline:", mse_mean)

visualizer = Visualizer()
visualizer.plot_preds(true, pred, test_dates)