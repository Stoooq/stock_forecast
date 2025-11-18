from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class StockDataGenerator(Dataset):
    def __init__(
            self, 
            csv_path: str,
            window_size: int = 30,
            test_size: float = 0.2,
            validation_size: float = 0.2,
            target_type: Literal['log_return', 'price', 'log_price'] = 'log_return',
            split: Literal['train', 'val', 'test'] = 'train',
            scaler = None,
    ):
        self.csv_path = csv_path #
        self.window_size = window_size #
        self.test_size = test_size #
        self.validation_size = validation_size #
        self.target_type = target_type #
        self.split = split #
        self.scaler = scaler #
        
        self.df = pd.read_csv(csv_path, parse_dates=["Date"]) #
        self.sequences = [] #

        self.prepare_data()

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:    
        X, y = self.sequences[idx]
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:    
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_volume'] = np.log(df['Volume'] + 1)

        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)

        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']

        return df

    def _create_sequence(
        self, data: np.ndarray, target_column_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size, target_column_index])

        return np.array(X), np.array(y)
    
    def _split_data_by_type(self, df: pd.DataFrame) -> pd.DataFrame:
        total_samples = len(df)
        
        test_samples = int(total_samples * self.test_size)
        val_samples = int(total_samples * self.validation_size)
        train_samples = total_samples - test_samples - val_samples
        
        train_end = train_samples
        val_start = train_end
        val_end = val_start + val_samples
        test_start = val_end
        
        if self.split == 'train':
            return df.iloc[:train_end].copy()
        elif self.split == 'val':
            return df.iloc[val_start:val_end].copy()
        elif self.split == 'test':
            return df.iloc[test_start:].copy()
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
    def _fit_scaler(self, data: np.ndarray, scaler_type: str = 'minmax') -> None:
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("Scaler type must be 'minmax' or 'standard'")
        
        self.scaler.fit(data)

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def to_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> DataLoader:
        def collate_fn(batch):
            try:
                # Bezpieczniejsze stackowanie
                X_list = [item[0] for item in batch]
                y_list = [item[1] for item in batch]
                
                X_batch = torch.from_numpy(np.stack(X_list)).float()
                y_batch = torch.from_numpy(np.stack(y_list)).float().squeeze()
                
                return X_batch, y_batch
            except Exception as e:
                print(f"Error in collate_fn: {e}")
                raise

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    def prepare_data(self, scaler_type: str = 'minmax') -> None:
        df = self._prepare_features(self.df.copy())

        df = df.dropna()

        df_split = self._split_data_by_type(df)

        exclude_cols = ['Date', 'Currency']
        no_scale_cols = ['log_return', 'log_volume']
        
        all_feature_cols = [col for col in df_split.columns if col not in exclude_cols]
        scale_cols = [col for col in all_feature_cols if col not in no_scale_cols]

        if self.split == 'train':
            self._fit_scaler(df_split[scale_cols].values, scaler_type=scaler_type)
            df_split[scale_cols] = self._transform_data(df_split[scale_cols].values)
            print(f"✓ Fitted {scaler_type} scaler on {len(scale_cols)} features")
        else:
            if self.scaler is None:
                raise ValueError(f"For split='{self.split}', you must provide scaler from train!")
            df_split[scale_cols] = self._transform_data(df_split[scale_cols].values)
    
        # feature_cols = [col for col in df_split.columns if col != 'Date']
        data = df_split[all_feature_cols].values

        if self.target_type == 'log_return':
            target_col = 'log_return'
        elif self.target_type == 'price':
            target_col = 'Close'
        else:
            if 'log_price' not in df_split.columns:
                df_split['log_price'] = np.log(df_split['Close'])
            target_col = 'log_price'
        
        target_idx = all_feature_cols.index(target_col)

        X, y = self._create_sequence(data, target_column_index=target_idx)
        self.sequences = [(X[i], y[i]) for i in range(len(X))]

    def get_scaler(self):
        return self.scaler

    @staticmethod
    def create_loaders(
        csv_path: str,
        window_size: int = 30,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        target_type: Literal['log_return', 'price', 'log_price'] = 'log_return',
        batch_size: int = 64,
        scaler_type: str = 'minmax',
        num_workers: int = 0
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_gen = StockDataGenerator(
            csv_path=csv_path,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split='train',
            scaler=None
        )
        train_loader = train_gen.to_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        val_gen = StockDataGenerator(
            csv_path=csv_path,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split='val',
            scaler=train_gen.get_scaler()
        )
        val_loader = val_gen.to_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        test_gen = StockDataGenerator(
            csv_path=csv_path,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split='test',
            scaler=train_gen.get_scaler()
        )
        test_loader = test_gen.to_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader