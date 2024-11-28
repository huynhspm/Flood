import json
import torch
import rootutils
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datasets import FloodDataset
from src.models import FloodModule

ckpt_path = "weights/training_2024.ckpt"
model: FloodModule = FloodModule.load_from_checkpoint(checkpoint_path=ckpt_path)
model.eval()

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

res = 0
for i in tqdm(range(len(dataset))):
    x, cond, y, time, hd = dataset[i]

    input = torch.tensor(x[None, :, :], device=model.device)
    cond = torch.tensor(cond[None, :, :], device=model.device)

    pred = model(input, cond)
    pred = pred[0, -1, 8].detach().cpu().item()

    gts[(i//8) * 4 + i % 8] += y[8]
    preds[(i//8) * 4 + i % 8] += pred
    times[(i//8) * 4 + i % 8] = time
    hon_dau[(i//8) * 4 + i % 8] += hd[0]

for i in range(len(gts[4:-4])):
    gts[i + 4] /= 2
    preds[i + 4] /= 2
    hon_dau[i + 4] /= 2

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

mean_hn, std_hn = metadata["mean"]["Hà Nội"], metadata["std"]["Hà Nội"]
print("Mean, std:", mean_hn, std_hn)

gts = [gt * std_hn + mean_hn for gt in gts]
preds = [pred * std_hn + mean_hn for pred in preds]

mean_hd, std_hd = metadata["mean"]["Hòn Dấu"], metadata["std"]["Hòn Dấu"]
diff = [pred - gt for pred, gt in zip(preds, gts)]
hon_dau = [hd * std_hd + mean_hd for hd in hon_dau]

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
# plt.plot(diff, label='Difference', color='brown')

plt.xlabel('time')
plt.ylabel('cm')
plt.title(f"2024 - MAE: {mae}")

plt.legend()
plt.savefig(f"result_{mode}_{mae}.jpg")


