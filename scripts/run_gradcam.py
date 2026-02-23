import torch
import argparse
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from collections import defaultdict
from torchvision.utils import save_image


from mst.models.dino import DinoV2ClassifierSlice
from mst_xai.xai_methods.gradcam_patch_level import GradCAM_MST
from mst_xai.xai_methods.gradcam_slice_level import GradCAM_Slice
from mst.data.datasets.dataset_3d_odelia import ODELIA_Dataset3D



parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', default='./runs', type=str)
parser.add_argument('--run_folder', required=True, type=str)
parser.add_argument('--output_dir', default='./', type=str)
parser.add_argument('--use_tta', action='store_true')
parser.add_argument('--max_importance', type=int, default=-1,
                    help='-1 = all test samples')
parser.add_argument('--max_images_per_class', type=int, default=5,
                    help='Images per class (qualitative only)')
parser.add_argument('--only_images', action='store_true',
                    help='Only generate images from saved importance')



args = parser.parse_args()


def get_dataset(name, split):
    if name == 'ODELIA':
        return ODELIA_Dataset3D(split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def overlay_heatmap(image, saliency, alpha=0.5):
    """
    image: 2D numpy array (H, W), normalized [0,1]
    saliency: 2D numpy array (H, W), normalized [0,1]
    """
    image_uint8 = np.uint8(255 * image)
    heatmap_uint8 = np.uint8(255 * saliency)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        heatmap_color,
        alpha,
        np.stack([image_uint8]*3, axis=-1),
        1 - alpha,
        0
    )

    return overlay


def concat_input_overlay(input_2d, overlay_rgb):
    """
    input_2d: (H, W) normalized [0,1]
    overlay_rgb: (H, W, 3) uint8 or float [0,255]
    """
    # convert input to RGB
    input_uint8 = np.uint8(255 * input_2d)
    input_rgb = np.stack([input_uint8]*3, axis=-1)

    # ensure same dtype
    if overlay_rgb.dtype != np.uint8:
        overlay_rgb = np.uint8(overlay_rgb)

    # concatenate horizontally
    combined = np.concatenate([input_rgb, overlay_rgb], axis=1)
    return combined

# -------------
# Define Result root
# -------------

run_folder = Path(args.run_folder)

dataset = run_folder.parent.name    # DUKE / LIDC / MRNet / Local
model_name = run_folder.name.split('_', 1)[0]

path_run = Path(args.run_dir) / run_folder

results_folder = 'results_tta' if args.use_tta else 'results'

path_out = Path(args.output_dir) / results_folder / run_folder
path_out.mkdir(parents=True, exist_ok=True)

# Grad-CAM root
gradcam_root = path_out / 'gradcam'
gradcam_root.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

MAX_IMP = args.max_importance
MAX_IMG = args.max_images_per_class


# -------------------------
# 1. Load trained model
# -------------------------

ModelClass = DinoV2ClassifierSlice
model = ModelClass.load_best_checkpoint(path_run)

model.to(device)
model.eval()
gradcam = GradCAM_MST(model)

# ------------ Load dataset ----------------
ds_test = get_dataset(dataset, split='test')
labels = ds_test.df[ds_test.LABEL].values
# print(set(labels))

# -------------
# Define Index map from sampling image
# -------------
if args.only_images:
    random.seed(42)

    indices_by_class = defaultdict(list)

    # collect indices (labels only, fast)
    for i, gt in enumerate(labels):
        indices_by_class[int(gt)].append(i)

    selected_indices = []
    for c, idxs in indices_by_class.items():
        if len(idxs) == 0:
            continue
        chosen = random.sample(
            idxs,
            k=min(MAX_IMG, len(idxs))
        )
        selected_indices.extend(chosen)

    tqdm.write(
        f"[Grad-CAM] Image-only mode: "
        f"{len(selected_indices)} samples selected "
        f"({MAX_IMG} per class)"
    )

if not args.only_images:
    selected_indices = list(range(len(ds_test)))


# -------------------------
# 2. Loop for importance
# -------------------------

image_counter = defaultdict(int)
importance_rows = []

processed = 0


if args.only_images:
    total = len(ds_test) #Don't look on max_importance
else:
    total = len(ds_test) if args.max_importance == -1 else min(len(ds_test), args.max_importance)

for idx in tqdm(
    selected_indices,
    desc="Grad-CAM images" if args.only_images else "Grad-CAM importance",
    ncols=100):

    sample = ds_test[idx]
    uid = sample["uid"]
    gt = int(sample["target"])
    batch = {
        'source': sample['source'].unsqueeze(0).to(device),
        'target': torch.tensor([gt]).to(device)
    }

    class_dir = gradcam_root / f'class_{gt}'
    pt_dir   = class_dir / "pt"
    npy_dir  = class_dir / "npy"
    image_dir = class_dir / "image"
    
    pt_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # ---------- SAVE IMPORTANCE (ALWAYS) ----------
    pt_path  = pt_dir  / f'{uid}_importance.pt'
    npy_path = npy_dir / f'{uid}_importance.npy'



 
    # ---------- NORMAL MODE (DEFAULT) ----------
    if not args.only_images:
    
        with torch.no_grad():
            pred = model(batch["source"]).argmax(dim=1).item()
    
        saliency = gradcam.generate(batch, target_class=pred)
    
        sal_np = saliency.detach().cpu().numpy()
        torch.save(saliency.detach().cpu(), pt_path)
        np.save(npy_path, sal_np)
    
        # CSV MUST be appended here — no continue above this
        importance_rows.append({
            'UID': uid,
            'dataset': dataset,
            'model': model_name,
            'xai_method': 'gradcam',
            'class': gt,
            'mean_importance': float(sal_np.mean()),
            'max_importance': float(sal_np.max()),
            'std_importance': float(sal_np.std()),
            'top_1pct_mean': float(sal_np[sal_np >= np.quantile(sal_np, 0.99)].mean()),
            'importance_path': str(pt_path)
        })
    
    # ---------- IMAGE-ONLY MODE ----------
    else:
        if not pt_path.exists():
            continue

    saliency = torch.load(pt_path, map_location=device)


    


    # ---------- SAVE IMAGES (LIMITED PER CLASS) ----------
    if image_counter[gt] < MAX_IMG:

       
        mid = saliency.shape[0] // 2

        save_image(
            batch['source'][0, 0, mid].cpu(),
            image_dir / f'input_{uid}.png',
            normalize=True
        )

        save_image(
            saliency[mid].unsqueeze(0),
            image_dir / f'gradcam_{uid}.png',
            normalize=True
        )

        # --- Prepare slice for overlay ---
        img_slice = batch['source'][0, 0, mid].detach().cpu().numpy()
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() + 1e-8)
        
        sal_slice = saliency[mid].detach().cpu().numpy()
        
        overlay = overlay_heatmap(
            img_slice,
            sal_slice,
            alpha=0.6
        )
        
        plt.imsave(
            image_dir / f'overlay_{uid}.png',
            overlay
        )

        combined = concat_input_overlay(
            input_2d=img_slice,
            overlay_rgb=overlay
        )
        
        plt.imsave(
            image_dir / f"input_overlay_{uid}.png",
            combined
        )


        image_counter[gt] += 1

    processed += 1

    if not args.only_images and args.max_importance != -1:
        if processed >= args.max_importance:
            break

#Stop when test with only some amount of files
df = pd.DataFrame(importance_rows)
df.to_csv(
    gradcam_root / 'gradcam_importance_summary.csv',
    index=False
)
