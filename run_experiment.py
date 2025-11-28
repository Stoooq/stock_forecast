import torch
import torch.nn as nn

from utils.config import Config
from data.data_module import DataModule
from models.model import MyModel
from training.checkpointer import Checkpointer
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.backtester import Backtester
from viz.visualizer import Visualizer

cfg = Config("/Users/miloszglowacki/Desktop/code/python/stock_forecast/config/config.yaml")
device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

data_module = DataModule(cfg.data)
train_loader, val_loader, test_loader, scaler, test_dates = data_module.get_loaders()

for x, _ in train_loader:
    print(x.shape)

model = MyModel(input_size=55, hidden_size=512, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

checkpointer = Checkpointer(cfg.training)

trainer = Trainer(model, loss_fn, optimizer, cfg.training, checkpointer)
history = trainer.train(train_loader, val_loader)

evaluator = Evaluator(model, cfg.training)
true, pred, res = evaluator.evaluate(test_loader)

true = data_module.inverse_transform_target(true)
pred = data_module.inverse_transform_target(pred)

backtester = Backtester(true, pred, test_dates)

backtester.generate_signals()
backtester.simulate_wallet()

visualizer = Visualizer()
visualizer.plot_preds(true, pred, test_dates)
