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
from mst_xai.xai_methods.gradcam import GradCAM_MST
from mst.data.datasets.dataset_3d_odelia import ODELIA_Dataset3D
from mst.inference.predictor import load_model


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ODELIA")
    parser.add_argument("--model_name", default="DinoV2ClassifierSlice")
    parser.add_argument("--xai_method", choices=["gradcam", "attention","random"], required=True)

    parser.add_argument("--run_dir", default="./runs")
    parser.add_argument("--run_folder", required=True)
    parser.add_argument("--checkpoint_name", default=None, help= "Specific checkpoint file name (e.g., best.chkpt)")

    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--use_tta", action="store_true")

    parser.add_argument("--mode", default="spatial", choices=['spatial', 'slice'])
    parser.add_argument("--cam_method", default="gradcam", choices=['gradcam', 'hires_cam'], help="Method for Grad-CAM variant to use")
    parser.add_argument("--norelu", action="store_true", help="Whether to skip ReLU in Grad-CAM (i.e., allow negative importance scores)")
    parser.add_argument("--attention_method", default="last_layer", choices=['last_layer','slice_weighted_rollout','grad_sam'])

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

    if args.xai_method == "attention":
        xai_name = args.attention_method
        xai_root = path_out / xai_name
    elif args.xai_method == "gradcam":
        xai_name = args.cam_method + ("_no_relu" if args.norelu else "")
        xai_root = path_out / xai_name
    elif args.xai_method == "random":
        xai_name = "random"
        xai_root = path_out / xai_name
    xai_root.mkdir(parents=True, exist_ok=True)

    return path_run, xai_root, xai_name


def load_dataset(name):
    if name == "ODELIA":
        return ODELIA_Dataset3D(split="test")
    raise ValueError(f"Unknown dataset: {name}")


def load_model_unified(args, path_run, device):
    model = load_model(
        model_name=args.model_name,
        checkpoint_path=path_run,
        device=device,
    )
    return model.to(device).eval()

def build_xai(args, model):
    if args.xai_method == "gradcam":
        name = f"{args.cam_method if not args.norelu else f'{args.cam_method}_no_relu'}"
        return GradCAM_MST(model,
                           cam_method=args.cam_method,
                           relu=not args.norelu
                           ), name

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

    if args.xai_method == "random":

        with torch.no_grad():
            logits = model(batch["source"])
            pred = logits.argmax(dim=1).item()

        _, _, D, H, W = batch["source"].shape
        saliency = torch.rand(
            D, H, W,
            device=batch["source"].device
        )

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


    elif args.xai_method == "attention": 

        # Single forward pass with attention storage
        logits = model(
            batch["source"],
            save_attn=True
        )

        pred = logits.argmax(dim=1).item()

        saliency = xai.generate(
            batch,
            logits=logits,
            target_class=pred,
        )

    else: # Grad-CAM and variants

        with torch.no_grad():
            logits = model(batch["source"])
            pred = logits.argmax(dim=1).item()

        saliency = xai.generate(
            batch,
            target_class=pred,
        )

    return saliency, pred



# ============================================================
# MAIN LOOP
# ============================================================

def run_pipeline(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_run, xai_root, xai_name = setup_paths(args)

    model = load_model_unified(args, path_run, device)

    if args.xai_method == "random":
        xai = None
        method_name = "random"
    else:
        xai, method_name = build_xai(args, model)

    ds = load_dataset(args.dataset)
    labels = ds.df[ds.LABEL].values

    values = list(range(len(labels)))

    rows = []
    # image_counter = defaultdict(int)

    for idx in tqdm(values):

        sample = ds[idx]
        uid = sample["uid"]
        gt = int(sample["target"])

        batch = {
            "source": sample["source"].unsqueeze(0).to(device),
            "target": torch.tensor([gt], device=device)
        }

        saliency, pred = generate_saliency(args, xai, model, batch)
        sal_np = saliency.detach().cpu().numpy()

        class_dir = xai_root / f"class_{gt}"
        pt_dir = class_dir / "pt"
        npy_dir = class_dir / "npy"
        # image_dir = class_dir / "image"

        pt_dir.mkdir(parents=True, exist_ok=True)
        npy_dir.mkdir(parents=True, exist_ok=True)
        # image_dir.mkdir(parents=True, exist_ok=True)

        pt_path = pt_dir / f"{uid}_importance.pt"
        npy_path = npy_dir / f"{uid}_importance.npy"



        torch.save(saliency.detach().cpu(), pt_path)
        np.save(npy_path, sal_np)

        row = {
            'UID': uid,
            'dataset': args.dataset,
            'model': args.model_name,
            'xai_method': xai_name,
            'gt_class': gt,
            'pred_class': pred,
            'mean_importance': float(sal_np.mean()),
            'max_importance': float(sal_np.max()),
            'std_importance': float(sal_np.std()),
            'top_1pct_mean': float(sal_np[sal_np >= np.quantile(sal_np, 0.99)].mean()),
            'importance_path': str(pt_path)
        }

        rows.append(row)


        # # -------- IMAGE --------
        # if image_counter[gt] < args.max_images_per_class:
        #     save_images(batch, saliency, uid, method_name, args, image_dir)
        #     image_counter[gt] += 1

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