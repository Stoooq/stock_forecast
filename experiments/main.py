import torch
import torch.nn as nn

from utils.config import Config
from data.data_module import DataModule
from models.model import MyModel
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from viz.visualizer import Visualizer

cfg = Config("/Users/miloszglowacki/Desktop/code/python/stock_forecast/config/config.yaml")
device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

data_module = DataModule(cfg.data)
train_loader, val_loader, test_loader, scaler = data_module.get_loaders()

model = MyModel(input_size=19, hidden_size=64, num_layers=1)

trainer = Trainer(model, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001), cfg.training)
history = trainer.train(train_loader, val_loader)

evaluator = Evaluator(model, cfg.training)
y, pred, res = evaluator.evaluate(test_loader)

visualizer = Visualizer()
visualizer.plot_preds(y, pred)