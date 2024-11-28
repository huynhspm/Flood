from typing import Literal, Any
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TideDataset(Dataset):

    data_file = "tide.csv"
    output_columns = [3]
    input_columns = [4, 5, 6]
    n_timestamp_pred = 8

    def __init__(self, 
                data_dir: str,
                length: int,
                step: int,
                mean: float = None,
                std: float = None,
                mode: Literal["train", "val", "test"] = "train",
                handle_missing_values: Literal["drop", "mean", "interpolate"] = "drop",
    ) -> None:
        self.length = length
        self.step = step

        data_path = os.path.join(data_dir, self.data_file)
        self.load_data(data_path, mode, handle_missing_values)
        self.normalize(mean, std)

    def load_data(self, data_path: str, mode: str, handle_missing_values: str) -> None:
        self.data = pd.read_csv(data_path)
        print(f"Data {mode}")

        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data["month"] = self.data["timestamp"].dt.month
        self.data["day"] = self.data.apply(
            lambda row: len(self.data[
                (self.data["timestamp"] >= pd.Timestamp(year=row["timestamp"].year, month=1, day=1)) &
                (self.data["timestamp"] < row["timestamp"])
            ]),
            axis=1
        )

        if mode == "train":
            self.data = self.data[~self.data["timestamp"].dt.year.isin([2023, 2024])]
        elif mode == "test":
            self.data = self.data[self.data["timestamp"].dt.year == 2024]
        elif mode == "val":
            self.data = self.data[self.data["timestamp"].dt.year == 2023]

        print(self.data.shape)

    def normalize(self, mean: float, std: float):
        if mean is None:
            self.mean, self.std = [], []
            for i in range(5):
                if i not in self.input_columns:
                    self.mean.append(None)
                    self.std.append(None)
                else:
                    self.mean.append(self.data.iloc[:, i].mean())
                    self.std.append(self.data.iloc[:, i].std())
        else:
            self.mean = mean
            self.std = std

        print("Normalize:", self.mean[3], self.std[4], self.mean[3], self.std[4])

        for i in range(5):
            if i in self.input_columns:
                self.data.iloc[:, i] = (self.data.iloc[:, i] - self.mean[i]) / self.std[i]

    def __len__(self) -> int:
        n = len(self.data)
        return ((n - self.length - self.n_timestamp_pred) // self.step + 1) * self.n_timestamp_pred

    def __getitem__(self, index) -> Any:
        i, next_i = index // self.n_timestamp_pred, index % self.n_timestamp_pred
        i *= self.step

        input = self.data.iloc[i: i + self.length, self.input_columns].to_numpy()
        output = self.data.iloc[i + self.length + next_i, self.output_columns].to_numpy()

        return input.astype(np.float32), output.astype(np.float32)

if __name__ == "__main__":
    dataset = TideDataset(data_dir="./data", 
                          length=28,
                          step=4,
                          mode="val")

    print('Length Dataset:', len(dataset))
    print(dataset.data.head(10))

    input, output = dataset[0]
    print(input.shape, output.shape)
    print(input.dtype, output.dtype)



