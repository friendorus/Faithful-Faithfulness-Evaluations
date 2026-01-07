import torch
import argparse
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torchvision.utils import save_image

from mst.models.dino import DinoV2ClassifierSlice
from mst_xai.xai_methods.attention import Attention_MST
from mst.data.datasets.dataset_3d_local import Local_Dataset3D


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', default='./runs', type=str)
parser.add_argument('--run_folder', required=True, type=str)
parser.add_argument('--output_dir', default='./', type=str)
parser.add_argument('--use_tta', action='store_true')
parser.add_argument('--mode', default='spatial', choices=['spatial', 'slice'])
parser.add_argument('--max_importance', type=int, default=-1)
parser.add_argument('--max_images_per_class', type=int, default=5)
parser.add_argument('--only_images', action='store_true')
args = parser.parse_args()


# ============================================================
# Dataset selector
# ============================================================

def get_dataset(name, split):
    if name == 'Local':
        return Local_Dataset3D(split=split)
    raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# Visualization helpers (same as Grad-CAM)
# ============================================================

def overlay_heatmap(image, saliency, alpha=0.5):
    image_uint8 = np.uint8(255 * image)
    heatmap_uint8 = np.uint8(255 * saliency)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(
        heatmap_color, alpha,
        np.stack([image_uint8]*3, axis=-1), 1 - alpha, 0
    )


def concat_input_overlay(input_2d, overlay_rgb):
    input_uint8 = np.uint8(255 * input_2d)
    input_rgb = np.stack([input_uint8]*3, axis=-1)
    return np.concatenate([input_rgb, overlay_rgb], axis=1)


# ============================================================
# Paths
# ============================================================

run_folder = Path(args.run_folder)
dataset = run_folder.parent.name
model_name = run_folder.name.split('_', 1)[0]

path_run = Path(args.run_dir) / run_folder
results_folder = 'results_tta' if args.use_tta else 'results'
path_out = Path(args.output_dir) / results_folder / run_folder
path_out.mkdir(parents=True, exist_ok=True)

attn_root = path_out / 'attention'
attn_root.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

MAX_IMP = args.max_importance
MAX_IMG = args.max_images_per_class


# ============================================================
# Model & Dataset
# ============================================================

model = DinoV2ClassifierSlice.load_best_checkpoint(path_run)
model.to(device).eval()

xai = Attention_MST(model, mode=args.mode)
ds_test = get_dataset(dataset, split='test')
labels = ds_test.df[ds_test.LABEL].values


# ============================================================
# Index selection (same logic as Grad-CAM)
# ============================================================

if args.only_images:
    random.seed(42)
    indices_by_class = defaultdict(list)
    for i, gt in enumerate(labels):
        indices_by_class[int(gt)].append(i)

    selected_indices = []
    for _, idxs in indices_by_class.items():
        selected_indices.extend(
            random.sample(idxs, k=min(MAX_IMG, len(idxs)))
        )
else:
    selected_indices = list(range(len(ds_test)))


# ============================================================
# Main loop
# ============================================================

image_counter = defaultdict(int)
importance_rows = []
processed = 0

for idx in tqdm(
    selected_indices,
    desc="Attention images" if args.only_images else "Attention importance",
    ncols=100
):
    sample = ds_test[idx]
    uid = sample["uid"]
    gt = int(sample["target"])

    batch = {
        "source": sample["source"].unsqueeze(0).to(device),
        "target": torch.tensor([gt], device=device)
    }

    class_dir = attn_root / f"class_{gt}"
    pt_dir = class_dir / "pt"
    npy_dir = class_dir / "npy"
    image_dir = class_dir / "image"
    pt_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pt_path = pt_dir / f"{uid}_importance.pt"
    npy_path = npy_dir / f"{uid}_importance.npy"

    # ---------------- NORMAL MODE ----------------
    if not args.only_images:
        saliency = xai.generate(batch, target_class=None)
        sal_np = saliency.detach().cpu().numpy()
        torch.save(saliency.cpu(), pt_path)
        np.save(npy_path, sal_np)

        importance_rows.append({
            "UID": uid,
            "dataset": dataset,
            "model": model_name,
            "xai_method": f"attention_{args.mode}",
            "class": gt,
            "mean_importance": float(sal_np.mean()),
            "max_importance": float(sal_np.max()),
            "std_importance": float(sal_np.std()),
            "importance_path": str(pt_path),
        })

    # ---------------- IMAGE-ONLY MODE ----------------
    else:
        if not pt_path.exists():
            continue
        saliency = torch.load(pt_path, map_location=device)

    # ---------------- SAVE IMAGES ----------------
    if image_counter[gt] < MAX_IMG and args.mode == "spatial":
        mid = saliency.shape[0] // 2

        save_image(
            batch["source"][0, 0, mid].cpu(),
            image_dir / f"input_{uid}.png",
            normalize=True
        )

        save_image(
            saliency[mid].unsqueeze(0),
            image_dir / f"attention_{uid}.png",
            normalize=True
        )

        img = batch["source"][0, 0, mid].cpu().numpy()
        img = (img - img.min()) / (img.max() + 1e-8)
        sal = saliency[mid].cpu().numpy()

        overlay = overlay_heatmap(img, sal)
        plt.imsave(image_dir / f"overlay_{uid}.png", overlay)

        combined = concat_input_overlay(img, overlay)
        plt.imsave(image_dir / f"input_overlay_{uid}.png", combined)

        image_counter[gt] += 1

    processed += 1
    if not args.only_images and MAX_IMP != -1 and processed >= MAX_IMP:
        break


# ============================================================
# Save CSV
# ============================================================

if not args.only_images:
    df = pd.DataFrame(importance_rows)
    df.to_csv(
        attn_root / f"attention_{args.mode}_summary.csv",
        index=False
    )
