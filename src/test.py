import json
import torch
import rootutils
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datasets import FloodDataset
from src.models import FloodModule, TideModule

ckpt_path = "weights/not_training_2024.ckpt"
model: FloodModule = FloodModule.load_from_checkpoint(checkpoint_path=ckpt_path)
model.eval()

tide_ckpt_path = "weights/tide_not_training_2024.ckpt"
tide_model: TideModule = TideModule.load_from_checkpoint(checkpoint_path=tide_ckpt_path)
tide_model.eval
tide_metadata_path = "weights/tide_metadata.json"

mode = "test"
metadata_path = "weights/metadata.json"
dataset = FloodDataset(data_dir="./data",
                        length=28,
                        step=4,
                        mode=mode,
                        metadata_path=metadata_path,
                        handle_missing_values="interpolate",
                        use_other_info=True)

print(len(dataset))
gts = [0 for _ in range(len(dataset) // 2 + 4)]
preds = [0 for _ in range(len(dataset) // 2 + 4)]
times = [None for _ in range(len(dataset) // 2 + 4)]
hon_dau = [0 for _ in range(len(dataset) // 2 + 4)]
pred_diffs = [0 for _ in range(len(dataset) // 2 + 4)]

res = 0
for i in tqdm(range(len(dataset))):
    x, cond, y, time, hd = dataset[i]

    input = torch.tensor(x[None, :, :], device=model.device)
    cond = torch.tensor(cond[None, :, :], device=model.device)

    pred = model(input, cond)
    pred = pred[0, -1, 8].detach().cpu().item()

    tide_input = cond[:, :, [1, 5, 6]]
    tide_pred = tide_model(tide_input)
    diff = tide_pred[:, -1, 0].detach().cpu().item()
    pred_diffs[(i//8) * 4 + i % 8] += diff

    gts[(i//8) * 4 + i % 8] += y[8]
    preds[(i//8) * 4 + i % 8] += pred
    times[(i//8) * 4 + i % 8] = time
    hon_dau[(i//8) * 4 + i % 8] += hd[0]

for i in range(len(gts[4:-4])):
    gts[i + 4] /= 2
    preds[i + 4] /= 2
    hon_dau[i + 4] /= 2
    pred_diffs[i + 4] /= 2

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)
mean_hn, std_hn = metadata["mean"]["Hà Nội"], metadata["std"]["Hà Nội"]
print("Mean, std:", mean_hn, std_hn)
gts = [gt * std_hn + mean_hn for gt in gts]
preds = [pred * std_hn + mean_hn for pred in preds]

diffs = [pred - gt for pred, gt in zip(preds, gts)]
mean_hd, std_hd = metadata["mean"]["Hòn Dấu"], metadata["std"]["Hòn Dấu"]
hon_dau = [hd * std_hd + mean_hd for hd in hon_dau]

with open(tide_metadata_path, "r", encoding="utf-8") as file:
    tide_metadata = json.load(file)
mean_diff, std_diff = tide_metadata["mean"]["diff"], tide_metadata["std"]["diff"]
print("Mean, std:", mean_diff, std_diff)
tide_diffs = [diff * std_diff + mean_diff for diff in pred_diffs]

preds = [pred - diff for pred, diff in zip(preds, pred_diffs)]

mae = 0
for pred, gt in zip(preds, gts):
    mae += abs(pred - gt)
mae /= len(gts)
print("MAE:", mae)

df = pd.DataFrame({
    "timestamp": times,
    "ground-truth": gts,
    "prediction": preds,
    "diff": diff,
    "hon_dau": hon_dau,
})

df.to_csv(f"result_{mode}_{mae}.csv", index=None)

plt.figure(figsize=(12, 6))
plt.plot(gts, label='Ground Truth', color='green')
plt.plot(preds, label='Predictions', color='yellow')

# plt.plot(hon_dau, label='Hòn Dấu', color='pink')
plt.plot(diffs, label='Difference', color='brown')
plt.plot(tide_diffs, label='Pred-Diff', color='purple')

plt.xlabel('time')
plt.ylabel('cm')
plt.title(f"2024 - MAE: {mae}")

plt.legend()
plt.savefig(f"result_{mode}_{mae}.jpg")


