from model import MyModel
import torch
import torch.nn as nn

from data_preparator import DataPreparator
from model_trainer import ModelTrainer

preparator = DataPreparator(
    data_url="/Users/miloszglowacki/Desktop/code/python/gold_forecast/data/gold.csv",
    window_size=30,
    test_size=0.2,
    validation_size=0.2,
)
df = preparator.load_data()
df_clean = preparator.clean_data(df, columns=["Date", "Currency"])

preparator.fit_scaler(df_clean.values)
data_scaled = preparator.transform_data(df_clean.values)

X, y = preparator.create_sequence(data_scaled, target_column_index=0)

X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_train_val_test(X, y)

X_train_t, y_train_t = preparator.to_tensor(X_train, y_train)
X_val_t, y_val_t = preparator.to_tensor(X_val, y_val)
X_test_t, y_test_t = preparator.to_tensor(X_test, y_test)

train_loader = preparator.create_dataloader(X_train_t, y_train_t, batch_size=64)
val_loader = preparator.create_dataloader(X_val_t, y_val_t, batch_size=64)
test_loader = preparator.create_dataloader(X_test_t, y_test_t, batch_size=64)

scaler = preparator.get_scaler()


model = MyModel(input_size=5, hidden_size=32, num_layers=1)
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

trainer.plot_predictions(test_loader, scaler)

print(train_history)
print(eval_history)