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
from mst_xai.xai_methods.gradcam_patch_level import GradCAM_MST
from mst.data.datasets.dataset_3d_odelia import ODELIA_Dataset3D
from mst.inference.predictor import load_model


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ODELIA")
    parser.add_argument("--model_name", default="DinoV2ClassifierSlice")
    parser.add_argument("--xai_method", choices=["gradcam", "attention"], required=True)

    parser.add_argument("--run_dir", default="./runs")
    parser.add_argument("--run_folder", required=True)
    parser.add_argument("--checkpoint_name", default=None, help= "Specific checkpoint file name (e.g., best.chkpt)")

    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--use_tta", action="store_true")

    parser.add_argument("--mode", default="spatial", choices=['spatial', 'slice'])
    parser.add_argument("--attention_method", default="last_layer", choices=['last_layer','slice_weighted_rollout'])

    parser.add_argument("--max_importance", type=int, default=-1,
                        help="Maximum number of input images to save importance scores(-1 for no limit)"
                        )
    parser.add_argument("--max_images_per_class", type=int, default=5,
                        help="Maximum number of images to save per class"
                        )
    parser.add_argument("--slice_choosen", default="squared_sum", 
                        choices=['mid', 'highest_mean', 'highest_max', 'top-K_mean', 'squared_sum'],
                        help="Method to select the slice for visualization when mode is spatial"
                        )
    parser.add_argument("--only_images", action="store_true",
                        help="If set, only saves images, not importance scores"
                        )

    return parser.parse_args()


# ============================================================
# SETUP
# ============================================================

def setup_paths(args):
    run_folder = Path(args.run_folder)

    path_run = Path(args.run_dir) / run_folder
    if args.checkpoint_name:
        path_run = path_run / args.checkpoint_name

    results_folder = "results_tta" if args.use_tta else "results"
    path_out = Path(args.output_dir) / results_folder / run_folder / "saliency_results"

    xai_root = path_out / args.xai_method
    xai_root.mkdir(parents=True, exist_ok=True)

    return path_run, xai_root


def load_dataset(name):
    if name == "ODELIA":
        return ODELIA_Dataset3D(split="test")
    raise ValueError(f"Unknown dataset: {name}")


def load_model_unified(args, path_run, device):
    # if args.xai_method == "attention":
    #     model = DinoV2ClassifierSlice.load_best_checkpoint(path_run)
    # else:
    model = load_model(
        model_name=args.model_name,
        checkpoint_path=path_run,
        device=device,
    )

    return model.to(device).eval()


def build_xai(args, model):
    if args.xai_method == "gradcam":
        return GradCAM_MST(model), "gradcam"

    if args.xai_method == "attention":
        name = f"{args.attention_method}_{args.mode}"
        return Attention_MST(
            model,
            mode=args.mode,
            attention_method=args.attention_method
        ), name


# ============================================================
# CORE FUNCTIONS
# ============================================================

def generate_saliency(args, xai, model, batch):
    if args.xai_method == "gradcam":
        with torch.no_grad():
            pred = model(batch["source"]).argmax(dim=1).item()
        return xai.generate(batch, target_class=pred)

    return xai.generate(batch, target_class=None)


def select_indices(labels, args):
    if not args.only_images:
        return list(range(len(labels)))

    random.seed(42)
    indices_by_class = defaultdict(list)

    for i, gt in enumerate(labels):
        indices_by_class[int(gt)].append(i)

    selected = []
    for idxs in indices_by_class.values():
        selected.extend(random.sample(idxs, k=min(len(idxs), args.max_images_per_class)))

    return selected


def select_slice(saliency, method):
    if method == "mid":
        return saliency.shape[0] // 2

    if method == "highest_mean":
        return saliency.mean(dim=(1, 2)).argmax().item()

    if method == "highest_max":
        return saliency.amax(dim=(1, 2)).argmax().item()

    if method == "top-K_mean":
        scores = [torch.topk(saliency[d].flatten(), 100).values.mean()
                  for d in range(saliency.shape[0])]
        return torch.stack(scores).argmax().item()

    if method == "squared_sum":
        return (saliency ** 2).sum(dim=(1, 2)).argmax().item()

    raise ValueError("Invalid slice selection")


# ============================================================
# VISUALIZATION
# ============================================================

def overlay_heatmap(image, saliency, alpha=0.5):
    image_uint8 = np.uint8(255 * image)
    heatmap_uint8 = np.uint8(255 * saliency)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(
        heatmap_color, alpha,
        np.stack([image_uint8]*3, axis=-1),
        1 - alpha, 0
    )

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


def save_images(batch, saliency, uid, method_name, args, image_dir):
    idx = select_slice(saliency, args.slice_choosen)

    img = batch["source"][0, 0, idx].detach().cpu().numpy()
    img = (img - img.min()) / (img.max() + 1e-8)
    sal = saliency[idx].detach().cpu().numpy()

    overlay = overlay_heatmap(img, sal)
    combined = concat_input_overlay(img, overlay)

    save_image(batch["source"][0, 0, idx].detach().cpu(),
               image_dir / f"original_{uid}.png", normalize=True)

    save_image(saliency[idx].detach().cpu().unsqueeze(0),
               image_dir / f"{method_name}_{uid}.png", normalize=True)

    plt.imsave(image_dir / f"overlay_{uid}.png", overlay)
    plt.imsave(image_dir / f"compare_{uid}.png", combined)


# ============================================================
# MAIN LOOP
# ============================================================

def run_pipeline(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_run, xai_root = setup_paths(args)

    model = load_model_unified(args, path_run, device)
    xai, method_name = build_xai(args, model)

    ds = load_dataset(args.dataset)
    labels = ds.df[ds.LABEL].values

    indices = select_indices(labels, args)

    rows = []
    image_counter = defaultdict(int)

    for idx in tqdm(indices):

        sample = ds[idx]
        uid = sample["uid"]
        gt = int(sample["target"])

        batch = {
            "source": sample["source"].unsqueeze(0).to(device),
            "target": torch.tensor([gt], device=device)
        }

        class_dir = xai_root / f"class_{gt}"
        pt_dir = class_dir / "pt"
        npy_dir = class_dir / "npy"
        image_dir = class_dir / "image"

        pt_dir.mkdir(parents=True, exist_ok=True)
        npy_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        pt_path = pt_dir / f"{uid}.pt"

        # -------- SALIENCY --------
        if not args.only_images:
            saliency = generate_saliency(args, xai, model, batch)
            sal_np = saliency.detach().cpu().numpy()

            torch.save(saliency.detach().cpu(), pt_path)
            np.save(npy_dir / f"{uid}.npy", sal_np)

            row = {
                "UID": uid,
                "xai_method": method_name,
                "class": gt,
                "mean": float(sal_np.mean()),
                "max": float(sal_np.max()),
                "std": float(sal_np.std())
            }

            rows.append(row)

        else:
            if not pt_path.exists():
                continue
            saliency = torch.load(pt_path, map_location=device, weights_only=True)

        # -------- IMAGE --------
        if image_counter[gt] < args.max_images_per_class:
            save_images(batch, saliency, uid, method_name, args, image_dir)
            image_counter[gt] += 1

    # -------- SAVE CSV --------
    if rows:
        pd.DataFrame(rows).to_csv(
            xai_root / f"{method_name}_summary.csv",
            index=False
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)