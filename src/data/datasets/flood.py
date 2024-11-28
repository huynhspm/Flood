from typing import Literal, Any
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FloodDataset(Dataset):

    data_file = "flood.xlsx"
    c = ["Lào cai", "Vụ Quang"]
    output_columns = ["Lào cai", "Vụ Quang", "Hà giang", "Bắc mê", "Vĩnh tuy", "Hàm Yên", "Tuyên Quang", \
                    "Sơn Tây", "Hà Nội", "Chũ", "Phả lại", "Cửa Cấm", "Ba lạt", "Phú thọ", "Ba Thá", "Đồ Nghi",	"Phú Lễ"]
    cond_columns = ["Yên bái", "Hòn Dấu", "Q xả Thác Bà", "Q xả Hoà Bình", "Q xả Tuyên Quang"] + ["month", "day"]
    input_columns = output_columns + ["month", "day"]

    def __init__(self, 
                data_dir: str,
                length: int,
                step: int,
                metadata_path: str,
                mode: Literal["train", "val", "test"] = "train",
                handle_missing_values: Literal["drop", "mean", "interpolate"] = "drop",
                use_other_info: bool = False,
    ) -> None:
        self.length = length
        self.step = step
        self.mode = mode
        self.use_other_info = use_other_info

        if self.mode == "train":
            self.metadata = {}
        else:
            with open(metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        data_path = os.path.join(data_dir, self.data_file)
        self.load_data(data_path, handle_missing_values)
        self.normalize()

        self.metadata["output_columns"] = self.output_columns
        self.metadata["cond_columns"] = self.cond_columns
        self.metadata["input_columns"] = self.input_columns

        if self.mode == "train":
            with open("weights/metadata.json", "w", encoding="utf-8") as file:
                json.dump(self.metadata, file, indent=4, ensure_ascii=False)

    def load_data(self, data_path: str, handle_missing_values: str) -> None:
        raw_data = pd.read_excel(data_path, skiprows=[0], parse_dates=[0], dtype=float)

        self.data = raw_data
        self.handle_missing_values(method=handle_missing_values)

        self.data.rename(columns={self.data.columns[0]: "timestamp"}, inplace=True)
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

    def handle_missing_values(self, method: str = "drop"):
        if self.mode == "train":
            missing_percentage = (self.data.isnull().sum() / len(self.data))
            remove_columns = missing_percentage[missing_percentage > 0.35].index.tolist()
            self.metadata["remove_columns"] = remove_columns

        self.cond_columns = [col for col in self.cond_columns if col not in self.metadata["remove_columns"]]
        self.input_columns = [col for col in self.input_columns if col not in self.metadata["remove_columns"]]
        self.output_columns = [col for col in self.output_columns if col not in self.metadata["remove_columns"]]

        print("Remove columns:", self.metadata["remove_columns"])
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

    def normalize(self):
        if self.mode == "train":
            self.metadata["mean"], self.metadata["std"] = {}, {}

            for col in self.data.columns:
                if col in ["timestamp", "month", "day"] + self.metadata["remove_columns"]:
                    self.metadata["mean"][col] = None
                    self.metadata["std"][col] = None
                else:
                    self.metadata["mean"][col] = self.data.loc[:, col].mean()
                    self.metadata["std"][col] = self.data.loc[:, col].std()

        print("Normalize:", self.metadata["mean"]["Hà Nội"], self.metadata["std"]["Hà Nội"])

        for col in self.data.columns:
            if self.metadata["mean"][col] is not None:
                self.data.loc[:, col] = (self.data.loc[:, col] - self.metadata["mean"][col]) / self.metadata["std"][col]

    def __len__(self) -> int:
        n = len(self.data)
        return ((n - self.length - 8) // self.step + 1) * 8

    def __getitem__(self, index) -> Any:
        i, next_i = index // 8, index % 8
        i *= self.step

        input = self.data.loc[i: i + self.length - 1, self.input_columns].to_numpy()
        cond = self.data. loc[i: i + self.length + next_i, self.cond_columns].to_numpy()

        output = self.data.loc[i + self.length + next_i, self.output_columns].to_numpy()
        cond = np.concatenate((np.zeros((7 - next_i, cond.shape[1])), cond), axis=0)
        time = self.data.iloc[i + self.length + next_i, 0]
        hon_dau = self.data.iloc[i + self.length + next_i, 19:20].to_numpy()

        if self.use_other_info:
            return input.astype(np.float32), cond.astype(np.float32), output.astype(np.float32), time, hon_dau

        return input.astype(np.float32), cond.astype(np.float32), output.astype(np.float32)

if __name__ == "__main__":
    dataset = FloodDataset(data_dir="./data", 
                            length=28,
                            step=4,
                            metadata_path="weights/metadata.json",
                            mode="train",
                            handle_missing_values="interpolate")

    print('Length Dataset:', len(dataset))
    print(dataset.data.head(10))

    input, cond, output = dataset[0]
    print(input.shape, cond.shape, output.shape)
    print(input.dtype, cond.dtype, output.dtype)



