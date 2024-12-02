from typing import Literal, Any
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TideDataset(Dataset):

    data_file = "tide.csv"
    output_columns = ["diff"]
    input_columns = ["hon_dau", "month", "day"]
    n_timestamp_pred = 8

    def __init__(self, 
                data_dir: str,
                length: int,
                step: int,
                metadata_path: str,
                mode: Literal["train", "val", "test"] = "train",
                handle_missing_values: Literal["drop", "mean", "interpolate"] = "drop",
    ) -> None:
        self.length = length
        self.step = step
        self.mode = mode

        if self.mode == "train":
            self.metadata = {}
        else:
            with open(metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        data_path = os.path.join(data_dir, self.data_file)
        self.load_data(data_path)
        self.normalize()

        self.metadata["output_columns"] = self.output_columns
        self.metadata["input_columns"] = self.input_columns

        if self.mode == "train":
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(self.metadata, file, indent=4, ensure_ascii=False)

    def load_data(self, data_path: str) -> None:
        self.data = pd.read_csv(data_path, parse_dates=[0], dtype=float)

        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data["month"] = self.data["timestamp"].dt.month
        self.data["day"] = self.data.apply(
            lambda row: len(self.data[
                (self.data["timestamp"] >= pd.Timestamp(year=row["timestamp"].year, month=1, day=1)) &
                (self.data["timestamp"] < row["timestamp"])
            ]),
            axis=1
        )

        print(f"Data {self.mode}")

        if self.mode == "train":
            self.data = self.data[~self.data["timestamp"].dt.year.isin([2023, 2024])]
        elif self.mode == "test":
            self.data = self.data[self.data["timestamp"].dt.year == 2024]
        elif self.mode == "val":
            self.data = self.data[self.data["timestamp"].dt.year == 2023]

        self.data.reset_index(drop=True, inplace=True)

    def normalize(self):
        if self.mode == "train":
            self.metadata["mean"], self.metadata["std"] = {}, {}

            for col in self.data.columns:
                if col in ["timestamp", "month", "day"]:
                    self.metadata["mean"][col] = None
                    self.metadata["std"][col] = None
                else:
                    self.metadata["mean"][col] = self.data.loc[:, col].mean()
                    self.metadata["std"][col] = self.data.loc[:, col].std()

        print("Normalize:", self.metadata["mean"]["diff"], self.metadata["std"]["diff"])

        for col in self.data.columns:
            if self.metadata["mean"][col] is not None:
                self.data.loc[:, col] = (self.data.loc[:, col] - self.metadata["mean"][col]) / self.metadata["std"][col]

    def __len__(self) -> int:
        n = len(self.data)
        return ((n - self.length - self.n_timestamp_pred) // self.step + 1) * self.n_timestamp_pred

    def __getitem__(self, index) -> Any:
        i, next_i = index // self.n_timestamp_pred, index % self.n_timestamp_pred
        i *= self.step

        input = self.data.loc[i: i + self.length + next_i, self.input_columns].to_numpy()
        output = self.data.loc[i + self.length + next_i, self.output_columns].to_numpy()

        input = np.concatenate((np.zeros((self.n_timestamp_pred - next_i - 1, input.shape[1])), input), axis=0)

        return input.astype(np.float32), output.astype(np.float32)

if __name__ == "__main__":
    dataset = TideDataset(data_dir="./data", 
                          length=28,
                          step=4,
                          metadata_path="weights/tide_metadata.json",
                          mode="train")

    print('Length Dataset:', len(dataset))
    print(dataset.data.head(64))

    input, output = dataset[0]
    print(input.shape, output.shape)
    print(input.dtype, output.dtype)



