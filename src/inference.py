import json
import torch
import argparse
import rootutils
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datasets import FloodDataset
from src.models import FloodModule

def normalize():
    pass

def load_data(data_path: str, metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    cond_columns = metadata["cond_columns"]
    input_columns = metadata["input_columns"]
    output_column = "Hà Nội"
    win_len = 28
    n_output = 8
    step = 4

    data = pd.read_excel(data_path, skiprows=[0], parse_dates=[0], dtype=float)
    print("Output Columns:", data.columns)

    data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)
    data["timestamp"] = data["timestamp"].replace(r'\.\d+$', '', regex=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')

    data["month"] = data["timestamp"].dt.month
    data["day"] = data.apply(
        lambda row: len(data[
            (data["timestamp"] >= pd.Timestamp(year=row["timestamp"].year, month=1, day=1)) &
            (data["timestamp"] < row["timestamp"])
        ]),
        axis=1
    )

    if data.isnull().any(axis=1).sum():
        data.interpolate(method="linear", inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)

    # normalize
    for col in data.columns:
        if metadata["mean"][col] is not None:
            data.loc[:, col] = (data.loc[:, col] - metadata["mean"][col]) / metadata["std"][col]

    inputs, conds, outputs, times = [], [], [], []
    for idx in range(0, len(data) - win_len - n_output + 1, 4):
        b_input, b_cond, b_output, b_time = [], [], [], []
        for i in range(n_output):

            input = data.loc[idx: idx + win_len - 1, input_columns].astype(np.float32).to_numpy()
            b_input.append(torch.tensor(input))

            cond = data.loc[idx: idx + win_len + i, cond_columns].astype(np.float32).to_numpy()
            cond = torch.cat((torch.zeros((n_output - i - 1, cond.shape[1])), torch.tensor(cond)), axis=0)
            b_cond.append(cond)

            output = data.loc[idx + win_len + i, output_column].astype(np.float32)
            b_output.append(output.item())

            b_time.append(data.loc[idx + win_len + i, "timestamp"])

        inputs.append(torch.stack(b_input, dim=0))
        conds.append(torch.stack(b_cond, dim=0))
        outputs.append(b_output)
        times.append(b_time)

    return inputs, conds, outputs, times, metadata["mean"][output_column], metadata["std"][output_column]

def infer(input_file: str, metadata_path: str, ckpt_path: str):
    model: FloodModule = FloodModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()

    inputs, conds, outputs, times, mean, std = load_data(data_path=input_file, metadata_path=metadata_path)

    print("Mean Std:", mean, std)

    gts = {}
    preds = {}

    for b_input, b_cond, b_output, b_time in tqdm(zip(inputs, conds, outputs, times), total=len(inputs)):
        b_pred = model(b_input.to(model.device), b_cond.to(model.device))
        b_pred = b_pred[:, -1, 8].detach().cpu().tolist()

        for pred, gt, time in zip(b_pred, b_output, b_time):
            
            if time not in gts.keys():
                gts[time] = [gt]
                preds[time] = [pred]
            else:
                gts[time].append(gt)
                preds[time].append(pred)

    mae = 0
    for time in gts.keys():
        gts[time] = sum(gts[time]) / len(gts[time])
        preds[time] = sum(preds[time]) / len(preds[time])

        # re-normalize
        gts[time] = gts[time] * std + mean
        preds[time] = preds[time] * std + mean

        mae += abs(gts[time] -  preds[time])
    mae /= len(gts.keys())
    print("MAE:", mae)

    gts, preds, timestamps= list(gts.values()), list(preds.values()), list(gts.keys())
    output_file = input_file.split('.')[0] + "_res"
    output = {
        "timestamp": timestamps,
        "ground-truth": gts,
        "prediction": preds,
    }

    df_output = pd.DataFrame(output)
    df_output.to_csv(f"{output_file}.csv", index=False)
    df_output.to_excel(f"{output_file}.xlsx", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(gts, label='Ground Truth', color='green')
    plt.plot(preds, label='Predictions', color='yellow')
    plt.xlabel('time')
    plt.ylabel('cm')
    plt.title(f"MAE: {mae}")
    plt.legend()
    plt.savefig(f"{output_file}.jpg")


if __name__ == "__main__":
    metadata_path = "weights/metadata.json"

    parser = argparse.ArgumentParser(description="Flood")
    parser.add_argument("--input", type=str, 
                        default="results/year.xlsx", 
                        help="Path to the input file (default: 'results/year.xlsx')")
    parser.add_argument("--ckpt", 
                        type=str, 
                        default="weights/not_training_2024.ckpt", 
                        help="Path to the checkpoint file (default: 'weights/not_training_2024.ckpt')")
    args = parser.parse_args()

    infer(input_file=args.input, 
            metadata_path=metadata_path, 
            ckpt_path=args.ckpt)