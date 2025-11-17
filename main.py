from model import MyModel
import torch
import torch.nn as nn
from stock_data_generator_v2 import StockDataGenerator
from model_trainer import ModelTrainer

train_loader, val_loader, test_loader = StockDataGenerator.create_loaders(
    csv_path='/Users/miloszglowacki/Desktop/code/python/stock_forecast/data/gold.csv',
    target_col='log_return',
    exclude_cols=['Date', 'Currency'],
    no_scale_cols=['log_return', 'log_volume']
)

model = MyModel(input_size=19, hidden_size=32, num_layers=1)
trainer = ModelTrainer(
    model=model,
    loss_fn=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

train_history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
)

eval_history = trainer.evaluate(test_loader)

trainer.plot_history()

# trainer.plot_predictions(test_loader, scaler)

# print(train_history)
# print(eval_history)