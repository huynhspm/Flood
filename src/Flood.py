from typing import List
import json
import torch
from torch import Tensor
from datetime import datetime
from importlib.resources import files

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models import FloodModule

def predict(inputs: Tensor, conds: Tensor):
    metadata_path = files("weights").joinpath("metadata.json")
    ckpt_path = files("weights").joinpath("not_training_2024.ckpt")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    input_columns = metadata["input_columns"][:-2]
    input_mean = [metadata["mean"][col] for col in input_columns]
    input_std = [metadata["std"][col] for col in input_columns]
    input_mean = torch.tensor(input_mean)[None, None, :]
    input_std = torch.tensor(input_std)[None, None, :]

    cond_columns = metadata["cond_columns"][:-2]
    cond_mean = [metadata["mean"][col] for col in cond_columns]
    cond_std = [metadata["std"][col] for col in cond_columns]
    cond_mean = torch.tensor(cond_mean)[None, None, :]
    cond_std = torch.tensor(cond_std)[None, None, :]

    # normalize
    inputs[:, :, :-2] = (inputs[:, :, :-2] - input_mean) / input_std
    conds[:, :, :-2] = (conds[:, :, :-2] - cond_mean) / cond_std

    model: FloodModule = FloodModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    preds = model(inputs.to(model.device), conds.to(model.device))

    output = preds.detach().cpu()
    output = output * input_std + input_mean

    return output[:, -1, 8].tolist()

def calculate(lao_cai, vu_quang, ha_giang, bac_me, vinh_tuy, ham_yen, tuyen_quang, son_tay, ha_noi, chu, pha_lai, phu_tho, \
            yen_bai, hon_dau, xa_thac_ba, xa_hoa_binh, xa_tuyen_quang, \
            timestamp: List[datetime]):

    """
    Summary:
    Args:
        - lao_cai, vu_quang, ha_giang, bac_me, vinh_tuy, ham_yen, tuyen_quang, son_tay, ha_noi, chu, pha_lai, phu_tho
            - Water level information for measuring stations
            - Data: List[int] with a length of 28 (7 days ago, 4 timestamps per day)

        - yen_bai, hon_dau, xa_thac_ba, xa_hoa_binh, xa_tuyen_quang
            - Water level information for discharge stations:
            - Data: List[int] with a length of 36 (9 days ago, 4 timestamps per day)

        - timestamp
            - Data: List[datetime.datetime] with a length of 36 (9 days ago, 4 timestamps per day)
            - Format: (%Y-%m-%d %H:%M), including year, month, day, hour, and minute 

    Returns:
        - output:
            - Water level information for the Ha Noi measuring station
            - Data: List[int] with a length of 8 (2 upcoming days, 4 timestamps per day)
    """

    input = [lao_cai, vu_quang, ha_giang, bac_me, vinh_tuy, ham_yen, tuyen_quang, son_tay, ha_noi, chu, pha_lai, phu_tho]
    cond = [yen_bai, hon_dau, xa_thac_ba, xa_hoa_binh, xa_tuyen_quang]

    for i, station in enumerate(input):
        assert len(station) == 28, f"Length of measuring station {i}-th is not 28"

    for i, station in enumerate(cond):
        assert len(station) == 36, f"Length of discharge station {i + len(input)}-th is not 36"


    assert(len(timestamp) == 36), "Length of timestamp is not 36"

    length = 28
    n_timestamp_pred = 8

    month = [time.month for time in timestamp]
    day = [(time - datetime(time.year, 1, 1)).days * 4 + (time.hour // 6) for time in timestamp]

    input += [month[:length], day[:length]]
    input = torch.stack([torch.tensor(x, dtype=torch.float32) for x in input], dim=1)
    input = input.reshape(length, -1)
    inputs = input.unsqueeze(0).repeat(n_timestamp_pred, 1, 1)

    cond += [month, day]
    cond = torch.stack([torch.tensor(x, dtype=torch.float32) for x in cond], dim=1)
    cond = cond.reshape(length + n_timestamp_pred, -1)

    conds = []
    for i in range(n_timestamp_pred):
        c = cond[: length + i + 1]
        padded_c = torch.cat((torch.zeros((n_timestamp_pred - i - 1, c.shape[1])), c), dim=0)
        conds.append(padded_c)
    conds = torch.stack(conds, dim=0)

    return predict(inputs, conds)

if __name__ == "__main__":

    import pandas as pd
    input_columns = ["Lào cai", "Vụ Quang", "Hà giang", "Bắc mê", "Vĩnh tuy", "Hàm Yên", "Tuyên Quang", \
                    "Sơn Tây", "Hà Nội", "Chũ", "Phả lại", "Phú thọ"]
    cond_columns = ["Yên bái", "Hòn Dấu", "Q xả Thác Bà", "Q xả Hoà Bình", "Q xả Tuyên Quang"]
    
    data_path = "results/week.xlsx"
    data = pd.read_excel(data_path, skiprows=[0], parse_dates=[0], dtype=float)

    input_list = [data[col][:28].tolist() for col in input_columns]
    cond_list = [data[col][:36].tolist() for col in cond_columns]
    timestamp = data.iloc[:36, 0].tolist()

    timestamp = [time.to_pydatetime() for time in timestamp]
    params_list = input_list + cond_list + [timestamp]

    output = calculate(*params_list)
    print(output)