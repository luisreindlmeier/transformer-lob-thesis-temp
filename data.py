"""Data loading and preprocessing for LOBSTER LOB data."""
import os
from typing import Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from lightning import LightningDataModule
import config as cfg


def lobster_load(
    path: str,
    horizon: int = 10,
    seq_size: int = cfg.SEQ_SIZE,
    all_features: bool = True,
    return_midprice: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    data = np.load(path)
    
    horizon_to_col = {10: 4, 20: 3, 50: 2, 100: 1}
    if horizon not in horizon_to_col:
        raise ValueError(f"Unsupported horizon: {horizon}")
    
    col = horizon_to_col[horizon]
    labels = data[seq_size - cfg.LEN_SMOOTH:, -col]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    
    if all_features:
        lob_features = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        order_features = data[:, :cfg.LEN_ORDER]
        lob_tensor = torch.from_numpy(lob_features).float()
        order_tensor = torch.from_numpy(order_features).float()
        inputs = torch.cat((lob_tensor, order_tensor), dim=1)
    else:
        inputs = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        inputs = torch.from_numpy(inputs).float()
    
    if return_midprice:
        lob_features = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        best_ask = lob_features[:, 0]
        best_bid = lob_features[:, 2]
        midprice = (best_ask + best_bid) / 2
        midprice = midprice[seq_size - cfg.LEN_SMOOTH:len(labels) + seq_size - cfg.LEN_SMOOTH]
        return inputs, labels, midprice
    
    return inputs, labels


def compute_returns(midprice: np.ndarray, horizon: int = 10) -> np.ndarray:
    returns = np.zeros(len(midprice))
    for i in range(len(midprice) - horizon):
        if midprice[i] > 0 and midprice[i + horizon] > 0:
            ratio = midprice[i + horizon] / midprice[i]
            if ratio > 0:
                returns[i] = np.log(ratio)
    return returns


class LOBDataset(TorchDataset):
    def __init__(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], seq_size: int = cfg.SEQ_SIZE) -> None:
        self.seq_size = seq_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).long()
        max_sequences = len(self.x) - seq_size + 1
        self.length = min(len(self.y), max_sequences)
        self.data = self.x

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i:i + self.seq_size, :].clone(), self.y[i].clone()


class LOBDataModule(LightningDataModule):
    def __init__(self, train_set: LOBDataset, val_set: LOBDataset, test_set: Optional[LOBDataset] = None,
                 batch_size: int = cfg.BATCH_SIZE, num_workers: int = 4) -> None:
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = train_set.data.device.type != cfg.DEVICE

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_set is None:
            return None
        return DataLoader(self.test_set, batch_size=self.batch_size * 4, shuffle=False,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)


def reset_indexes(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes


def z_score_orderbook(data: pd.DataFrame, mean_size: Optional[float] = None, mean_prices: Optional[float] = None,
                      std_size: Optional[float] = None, std_prices: Optional[float] = None
                      ) -> Tuple[pd.DataFrame, float, float, float, float]:
    if mean_size is None or std_size is None:
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    if mean_prices is None or std_prices is None:
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    for col in data.columns[1::2]:
        data[col] = (data[col].astype("float64") - mean_size) / std_size
    for col in data.columns[0::2]:
        data[col] = (data[col].astype("float64") - mean_prices) / std_prices

    return data, mean_size, mean_prices, std_size, std_prices


def normalize_messages(data: pd.DataFrame, mean_size: Optional[float] = None, mean_prices: Optional[float] = None,
                       std_size: Optional[float] = None, std_prices: Optional[float] = None,
                       mean_time: Optional[float] = None, std_time: Optional[float] = None,
                       mean_depth: Optional[float] = None, std_depth: Optional[float] = None
                       ) -> Tuple[pd.DataFrame, float, float, float, float, float, float, float, float]:
    if mean_size is None: mean_size = data["size"].mean()
    if std_size is None: std_size = data["size"].std()
    if mean_prices is None: mean_prices = data["price"].mean()
    if std_prices is None: std_prices = data["price"].std()
    if mean_time is None: mean_time = data["time"].mean()
    if std_time is None: std_time = data["time"].std()
    if mean_depth is None: mean_depth = data["depth"].mean()
    if std_depth is None: std_depth = data["depth"].std()

    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    # Remap event types: 1->0, 3->1, 4->2
    data["event_type"] = data["event_type"] - 1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)

    return data, mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth


def labeling(X: np.ndarray, len_smooth: int, h: int) -> np.ndarray:
    # Labels: 0=up, 1=stationary, 2=down (exact replica of original TLOB)
    if h < len_smooth:
        len_smooth = h
    
    previous_ask = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len_smooth)[:-h]
    previous_bid = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len_smooth)[:-h]
    future_ask = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len_smooth)[h:]
    future_bid = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len_smooth)[h:]

    previous_mid = np.mean((previous_ask + previous_bid) / 2, axis=1)
    future_mid = np.mean((future_ask + future_bid) / 2, axis=1)

    pct_change = (future_mid - previous_mid) / previous_mid
    alpha = np.abs(pct_change).mean() / 2

    return np.where(pct_change < -alpha, 2, np.where(pct_change > alpha, 0, 1))


class LOBSTERPreprocessor:
    COLUMNS_NAMES = {
        "orderbook": ["sell1", "vsell1", "buy1", "vbuy1", "sell2", "vsell2", "buy2", "vbuy2",
                      "sell3", "vsell3", "buy3", "vbuy3", "sell4", "vsell4", "buy4", "vbuy4",
                      "sell5", "vsell5", "buy5", "vbuy5", "sell6", "vsell6", "buy6", "vbuy6",
                      "sell7", "vsell7", "buy7", "vbuy7", "sell8", "vsell8", "buy8", "vbuy8",
                      "sell9", "vsell9", "buy9", "vbuy9", "sell10", "vsell10", "buy10", "vbuy10"],
        "message": ["time", "event_type", "order_id", "size", "price", "direction"]
    }
    
    def __init__(self, raw_data_dir: str, output_dir: str, split_rates: Optional[List[float]] = None) -> None:
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.split_rates = split_rates or cfg.SPLIT_RATES
        self.n_lob_levels = cfg.N_LOB_LEVELS
        self.dataframes: List[List[pd.DataFrame]] = []
    
    def preprocess(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')])
        self.num_trading_days = len(csv_files) // 2
        
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(split_days)
        
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"].astype(float) / 10000
            price_cols = self.dataframes[i][1].columns[::2]
            self.dataframes[i][1][price_cols] = self.dataframes[i][1][price_cols].astype(float) / 10000
        
        train_input = self.dataframes[0][1].values
        val_input = self.dataframes[1][1].values
        test_input = self.dataframes[2][1].values
        
        for i, h in enumerate(cfg.LOBSTER_HORIZONS):
            train_labels = labeling(train_input, cfg.LEN_SMOOTH, h)
            val_labels = labeling(val_input, cfg.LEN_SMOOTH, h)
            test_labels = labeling(test_input, cfg.LEN_SMOOTH, h)
            
            train_labels = np.concatenate([train_labels, np.full(train_input.shape[0] - len(train_labels), np.inf)])
            val_labels = np.concatenate([val_labels, np.full(val_input.shape[0] - len(val_labels), np.inf)])
            test_labels = np.concatenate([test_labels, np.full(test_input.shape[0] - len(test_labels), np.inf)])
            
            if i == 0:
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=[f"label_h{h}"])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=[f"label_h{h}"])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=[f"label_h{h}"])
            else:
                self.train_labels_horizons[f"label_h{h}"] = train_labels
                self.val_labels_horizons[f"label_h{h}"] = val_labels
                self.test_labels_horizons[f"label_h{h}"] = test_labels
        
        self._normalize_dataframes()
        self._save()
        print(f"Preprocessing complete. Output saved to {self.output_dir}")
    
    def _split_days(self) -> List[int]:
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"Train: {train} days, Val: {val - train} days, Test: {test - val} days")
        return [train, val, test]
    
    def _create_dataframes_splitted(self, split_days: List[int]) -> None:
        csv_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')])
        train_messages = train_orderbooks = val_messages = val_orderbooks = test_messages = test_orderbooks = None
        
        for i, filename in enumerate(csv_files):
            filepath = os.path.join(self.raw_data_dir, filename)
            is_message = (i % 2) == 0
            
            if i < split_days[0]:
                if is_message:
                    if i == 0:
                        train_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        train_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == 1:
                        train_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        train_orderbooks, train_messages = self._preprocess_message_orderbook([train_messages, train_orderbooks])
                    else:
                        train_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        train_orderbook, train_message = self._preprocess_message_orderbook([train_message, train_orderbook])
                        train_messages = pd.concat([train_messages, train_message], axis=0)
                        train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)
            
            elif split_days[0] <= i < split_days[1]:
                if is_message:
                    if i == split_days[0]:
                        self.dataframes.append([train_messages, train_orderbooks])
                        val_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        val_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == split_days[0] + 1:
                        val_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        val_orderbooks, val_messages = self._preprocess_message_orderbook([val_messages, val_orderbooks])
                    else:
                        val_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        val_orderbook, val_message = self._preprocess_message_orderbook([val_message, val_orderbook])
                        val_messages = pd.concat([val_messages, val_message], axis=0)
                        val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)
            
            else:
                if is_message:
                    if i == split_days[1]:
                        self.dataframes.append([val_messages, val_orderbooks])
                        test_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        test_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == split_days[1] + 1:
                        test_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        test_orderbooks, test_messages = self._preprocess_message_orderbook([test_messages, test_orderbooks])
                    else:
                        test_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        test_orderbook, test_message = self._preprocess_message_orderbook([test_message, test_orderbook])
                        test_messages = pd.concat([test_messages, test_message], axis=0)
                        test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)
        
        self.dataframes.append([test_messages, test_orderbooks])
    
    def _preprocess_message_orderbook(self, dataframes: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataframes = reset_indexes(dataframes)
        dataframes[1] = dataframes[1].iloc[:, :self.n_lob_levels * cfg.LEN_LEVEL]
        
        indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
        dataframes[0] = dataframes[0].drop(indexes_to_drop)
        dataframes[1] = dataframes[1].drop(indexes_to_drop)
        dataframes = reset_indexes(dataframes)
        
        dataframes[0] = dataframes[0].drop(columns=["order_id"])
        first_time = dataframes[0]["time"].values[0]
        dataframes[0]["time"] = dataframes[0]["time"].diff()
        dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = first_time - 34200
        dataframes[0]["depth"] = 0
        
        prices = dataframes[0]["price"].values
        directions = dataframes[0]["direction"].values
        event_types = dataframes[0]["event_type"].values
        bid_sides = dataframes[1].iloc[:, 2::4].values
        ask_sides = dataframes[1].iloc[:, 0::4].values
        
        depths = np.zeros(dataframes[0].shape[0], dtype=int)
        for j in range(1, len(prices)):
            index = j if event_types[j] == 1 else j - 1
            if directions[j] == 1:
                depth = (bid_sides[index, 0] - prices[j]) // 100
            else:
                depth = (prices[j] - ask_sides[index, 0]) // 100
            depths[j] = max(depth, 0)
        
        dataframes[0]["depth"] = depths
        dataframes[0] = dataframes[0].iloc[1:, :]
        dataframes[1] = dataframes[1].iloc[1:, :]
        dataframes = reset_indexes(dataframes)
        
        dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(lambda x: -1 if x == 4 else 1)
        return dataframes[1], dataframes[0]
    
    def _normalize_dataframes(self) -> None:
        for i in range(len(self.dataframes)):
            if i == 0:
                self.dataframes[i][1], mean_size_ob, mean_prices_ob, std_size_ob, std_prices_ob = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size_ob, mean_prices_ob, std_size_ob, std_prices_ob)
        
        for i in range(len(self.dataframes)):
            if i == 0:
                self.dataframes[i][0], mean_size_msg, mean_prices_msg, std_size_msg, std_prices_msg, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size_msg, mean_prices_msg, std_size_msg, std_prices_msg, mean_time, std_time, mean_depth, std_depth)
    
    def _save(self) -> None:
        for i, name in enumerate(["train", "val", "test"]):
            messages, orderbook = self.dataframes[i]
            labels = [self.train_labels_horizons, self.val_labels_horizons, self.test_labels_horizons][i]
            messages = messages.reset_index(drop=True)
            orderbook = orderbook.reset_index(drop=True)
            input_data = pd.concat([messages, orderbook], axis=1)
            combined = pd.concat([pd.DataFrame(input_data.values), pd.DataFrame(labels.values)], axis=1).values
            output_path = os.path.join(self.output_dir, f"{name}.npy")
            np.save(output_path, combined)
            print(f"  Saved {output_path}: shape {combined.shape}")
