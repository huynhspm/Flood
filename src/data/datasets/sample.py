from typing import Literal, Any
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SampleDataset(Dataset):

    data_file = "sample.xlsx"

    def __init__(self, 
                data_dir: str,
                length: int,
                step: int,
                handle_missing_values: Literal["drop", "mean", "interpolate"] = "drop"
    ) -> None:
        self.length = length
        self.step = step

        data_path = os.path.join(data_dir, self.data_file)
        self.load_data(data_path)
        self.handle_missing_values(method=handle_missing_values)

    def load_data(self, data_path: str) -> None:
        raw_data = pd.read_excel(data_path, skiprows=[1])
        self.data = raw_data.drop(raw_data.columns[1], axis=1)
        self.data.columns = ["time", "output"] + ["next" + str(i) for i in range(6)] + ["pass" + str(i) for i in range(50)]
        self.next_columns = [i + 2 for i in range(6)]
        self.pass_columns = [i + 8 for i in range(50)]

        print(self.data.head())
        print("Shape:", self.data.shape)

    def handle_missing_values(self, method: str = "drop"):
        print("Missing:", self.data.isnull().any(axis=1).sum())

        if method == "drop":
            self.data.dropna(inplace=True)
        elif method == "mean":
            self.data.fillna(self.data.mean(), inplace=True)
        elif method == "interpolate":
            self.data.interpolate(method="linear", inplace=True)
            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)
        else:
            raise NotImplementedError(f"Not implemeted {method}")

        print("Shape after handle missing values:", self.data.shape)

    def __len__(self) -> int:
        n = len(self.data)
        return ((n - self.length - 8) // self.step + 1) * 8

    def __getitem__(self, index) -> Any:
        i, next_i = index // 8, index % 8 + 1
        input = self.data.iloc[i: i + self.length, self.pass_columns].to_numpy()
        cond = self.data.iloc[i: i + self.length + next_i, self.next_columns].to_numpy()
        output = self.data.loc[i + self.length + next_i, "output"]

        if next_i < 8:
            cond = np.concatenate((cond, np.zeros((8 - next_i, cond.shape[1]))), axis=0)

        return input.astype(np.float32), cond.astype(np.float32), output.astype(np.float32)

if __name__ == "__main__":
    dataset = SampleDataset(data_dir="./data", 
                            length=28,
                            step=4,
                            handle_missing_values="interpolate")

    print('Length Dataset:', len(dataset))
    print(dataset.data.head())

    input, cond, output = dataset[0]
    print(input.shape, cond.shape, output.shape)



