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
    parser.add_argument("--xai_method", choices=["gradcam", "attention"], required=True)

    parser.add_argument("--run_dir", default="./runs")
    parser.add_argument("--run_folder", required=True)
    parser.add_argument("--checkpoint_name", default=None, help= "Specific checkpoint file name (e.g., best.chkpt)")

    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--use_tta", action="store_true")

    parser.add_argument("--mode", default="spatial", choices=['spatial', 'slice'])
    parser.add_argument("--cam_method", default="gradcam", choices=['gradcam', 'hires_cam'], help="Method for Grad-CAM variant to use")
    parser.add_argument("--norelu", action="store_true", help="Whether to skip ReLU in Grad-CAM (i.e., allow negative importance scores)")
    parser.add_argument("--attention_method", default="last_layer", choices=['last_layer','slice_weighted_rollout','grad_sam'])

    # parser.add_argument("--max_importance", type=int, default=-1,
    #                     help="Maximum number of input images to save importance scores(-1 for no limit)"
    #                     )
    # parser.add_argument("--max_images_per_class", type=int, default=5,
    #                     help="Maximum number of images to save per class"
    #                     )
    # parser.add_argument("--slice_choosen", default="squared_sum", 
    #                     choices=['mid', 'highest_mean', 'highest_max', 'top-K_mean', 'squared_sum'],
    #                     help="Method to select the slice for visualization when mode is spatial"
    #                     )

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

# # --------------------------------------------------
# # Detect number of extra tokens (CLS + register/storage)
# # --------------------------------------------------
# def _get_num_extra_tokens(self, source):
#     """
#     This method detects how many extra tokens (beyond the CLS token) are present in the ViT encoder's output.        
#     # Remove CLS + extra tokens (we only want patch tokens for spatial saliency)
#     # ViT tokes = [CLS] + [extra tokens] + [patch tokens]
#     # For DinoV2, there are possible to have only 1 CLS toke or extra register tokens,
#     # For DinoV3, there are possible to have 1 CLS token + 4 storage tokens
#     """
#     if hasattr(self, "num_extra_tokens"):
#         return self.num_extra_tokens

#     # No gradient needed -> just inspect model structure to determine how many extra tokens there are (e.g., CLS + storage tokens)
#     with torch.no_grad():
#         # Take one slice for probing
#         x_enc = source[:1]              # (1,1,D,H,W)
#         x_enc = x_enc[:, :, 0]          # take one slice → (1,1,H,W)
#         # Convert to 3-channel by repeating the single channel (ViT requires 3-channel input) → (1,3,H,W)
#         x_enc = x_enc.repeat(1, 3, 1, 1)  # → (1,3,H,W)

#         # Get token structure
#         out = self.model.encoder.forward_features(x_enc)

#     # Detect exttra tokens based on the output of the encoder's forward_features method.
#     # If new models have different token structures, this logic may need to be updated.
#     if "x_storage_tokens" in out:
#         # DinoV3 (CLS + storage tokens)
#         self.num_extra_tokens = out["x_storage_tokens"].shape[1] 
#     elif "x_norm_regtokens" in out:
#         # DinoV2 (CLS + normalized register tokens)
#         self.num_extra_tokens = out["x_norm_regtokens"].shape[1]
#     else:
#         # Default to 0 if no extra tokens are detected (only CLS token)
#         self.num_extra_tokens = 0

#     return self.num_extra_tokens


# ============================================================
# CORE FUNCTIONS
# ============================================================

def generate_saliency(args, xai, model, batch):

    if args.xai_method == "attention": # and args.attention_method == "grad_sam":

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

    else:

        with torch.no_grad():
            pred = model(batch["source"]).argmax(dim=1).item()

        saliency = xai.generate(
            batch,
            target_class=pred,
        )

    return saliency, pred


# def select_indices(labels, args):
#     # if not args.only_images:
#     #     return list(range(len(labels)))

#     random.seed(42)
#     indices_by_class = defaultdict(list)

#     for i, gt in enumerate(labels):
#         indices_by_class[int(gt)].append(i)

#     selected = []
#     for idxs in indices_by_class.values():
#         selected.extend(random.sample(idxs, k=min(len(idxs), args.max_images_per_class)))

#     return selected


# def select_slice(saliency, method):
#     if method == "mid":
#         return saliency.shape[0] // 2

#     if method == "highest_mean":
#         return saliency.mean(dim=(1, 2)).argmax().item()

#     if method == "highest_max":
#         return saliency.amax(dim=(1, 2)).argmax().item()

#     if method == "top-K_mean":
#         scores = [torch.topk(saliency[d].flatten(), 100).values.mean()
#                   for d in range(saliency.shape[0])]
#         return torch.stack(scores).argmax().item()

#     if method == "squared_sum":
#         return (saliency ** 2).sum(dim=(1, 2)).argmax().item()

#     raise ValueError("Invalid slice selection")


# ============================================================
# VISUALIZATION
# ============================================================

# def overlay_heatmap(image, saliency, alpha=0.5):
#     """
#     image: 2D numpy array (H, W), normalized [0,1]
#     saliency: 2D numpy array (H, W), normalized [0,1]
#     """
#     image_uint8 = np.uint8(255 * image)
#     heatmap_uint8 = np.uint8(255 * saliency)

#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

#     overlay = cv2.addWeighted(
#         heatmap_color,
#         alpha,
#         np.stack([image_uint8]*3, axis=-1),
#         1 - alpha,
#         0
#     )

#     return overlay

# def concat_input_overlay(input_2d, overlay_rgb):
#     """
#     input_2d: (H, W) normalized [0,1]
#     overlay_rgb: (H, W, 3) uint8 or float [0,255]
#     """
#     # convert input to RGB
#     input_uint8 = np.uint8(255 * input_2d)
#     input_rgb = np.stack([input_uint8]*3, axis=-1)

#     # ensure same dtype
#     if overlay_rgb.dtype != np.uint8:
#         overlay_rgb = np.uint8(overlay_rgb)

#     # concatenate horizontally
#     combined = np.concatenate([input_rgb, overlay_rgb], axis=1)
#     return combined


# def save_images(batch, saliency, uid, method_name, args, image_dir):
#     img = batch["source"].squeeze(0).squeeze(0).cpu().numpy()
#     img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0,1]
#     idx = select_slice(saliency, args.slice_choosen)

#     img = img[idx]
#     sal = saliency[idx].detach().cpu().numpy()

#     overlay = overlay_heatmap(img, sal)
#     combined = concat_input_overlay(img, overlay)

#     save_image(batch["source"][0, 0, idx].detach().cpu(),
#                image_dir / f"original_{uid}.png", normalize=True)

#     save_image(saliency[idx].detach().cpu().unsqueeze(0),
#                image_dir / f"{method_name}_{uid}.png", normalize=True)

#     plt.imsave(image_dir / f"overlay_{uid}.png", overlay)
#     plt.imsave(image_dir / f"compare_{uid}.png", combined)


# ============================================================
# MAIN LOOP
# ============================================================

def run_pipeline(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_run, xai_root, xai_name = setup_paths(args)

    model = load_model_unified(args, path_run, device)
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