# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from model import MyModel
# from prepare_data import create_sequences
# import torch.nn as nn
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
from data_preparator import DataPreparator

preparator = DataPreparator(data_url="/Users/miloszglowacki/Desktop/code/python/gold_forecast/data/gold.csv", window_size=30, test_size=0.2, validation_size=0.2)
df = preparator.load_data()
df_clean = preparator.clean_data(df, columns=["Date", "Currency"])

preparator.fit_scaler(df_clean.values)
data_scaled = preparator.transform_data(df_clean.values)

X, y = preparator.create_sequence(data_scaled, target_column_index=0)

X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_train_val_test(X, y)

X_train_t, y_train_t = preparator.to_tensor(X_train, y_train)

train_loader = preparator.create_dataloader(X_train_t, y_train_t, batch_size=64)

# df = pd.read_csv(
#     "/Users/miloszglowacki/Desktop/code/python/gold_forecast/data/gold.csv",
#     parse_dates=["Date"],
# )
# df = df.iloc[-5000:]
# df = df.drop(columns=["Date", "Currency"])

# scaler = MinMaxScaler()
# df_scaled = scaler.fit_transform(df.values)

# X, y = create_sequences(df_scaled, 30)
# print(X.shape, y.shape)

# X_train = X[:-100]
# y_train = y[:-100]
# X_test = X[-100:]
# y_test = y[-100:]

# X_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_tensor = torch.tensor(y_train, dtype=torch.float32)

# dataset = TensorDataset(X_tensor, y_tensor)
# loader = DataLoader(dataset, batch_size=64, shuffle=True)

# model = MyModel(input_size=5, hidden_size=32, num_layers=1)
# loss_fn = nn.MSELoss()
# optim = torch.optim.Adam(model.parameters(), lr=0.001)

# loss_history = []
# for epoch in range(100):
#     epoch_loss = 0.0
#     for X_batch, y_batch in loader:
#         output = model(X_batch)
#         loss = loss_fn(output, y_batch)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         epoch_loss += loss.item()
#     avg_loss = loss / len(loader)
#     loss_history.append(avg_loss)

#     if epoch % 5 == 0:
#         print(f"Epoch: {epoch}, loss: {avg_loss}")

# model.eval()
# with torch.no_grad():
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#     test_predictions = model(X_test_tensor)
#     test_loss = loss_fn(test_predictions, y_test_tensor)
#     print(f"\nTest loss: {test_loss.item()}")

#     test_predictions_np = test_predictions.numpy()
#     y_test_np = y_test_tensor.numpy()

#     test_pred_full = np.zeros((test_predictions_np.shape[0], 5))
#     test_pred_full[:, 0] = test_predictions_np.squeeze()
    
#     y_test_full = np.zeros((y_test_np.shape[0], 5))
#     y_test_full[:, 0] = y_test_np.squeeze()
    
#     test_pred_original = scaler.inverse_transform(test_pred_full)[:, 0]
#     y_test_original = scaler.inverse_transform(y_test_full)[:, 0]

# plt.figure(figsize=(12, 6))
# plt.plot(y_test_original, label="Prawdziwa wartość", color="blue", linewidth=2)
# plt.plot(
#     test_pred_original, label="Predykcja", color="red", linewidth=2, linestyle="--"
# )
# plt.xlabel("Indeks próbki testowej")
# plt.ylabel("Cena złota")
# plt.title("Predykcje modelu vs Prawdziwe wartości (dane testowe)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("predictions_vs_actual.png")
# plt.show()

# print(f"\nWykres zapisany jako 'predictions_vs_actual.png'")
