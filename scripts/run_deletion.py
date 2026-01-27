import argparse
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tqdm import tqdm

from mst.models.dino import DinoV2ClassifierSlice
from mst.data.datasets.dataset_3d_odelia import ODELIA_Dataset3D

from mst_xai.evaluation.deletion import deletion_evaluation
from mst_xai.utils.load_saliency import load_saliency


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(description="Deletion faithfulness evaluation")
parser.add_argument("--run_dir", default="./runs", type=str)
parser.add_argument("--run_folder", required=True, type=str)
parser.add_argument("--output_dir", default="./", type=str)

parser.add_argument(
    "--xai_method",
    required=True,
    choices=["attention", "attention_rollout", "gradcam", "gmar"],
    help="Which saliency folder to evaluate",
)

parser.add_argument("--max_samples", type=int, default=-1,
                    help="-1 = all available saliency files")
parser.add_argument("--steps", type=int, default=20,
                    help="Number of deletion steps")
parser.add_argument("--replacement", default="zero",
                    choices=["zero", "mean"],
                    help="Replacement strategy")

parser.add_argument("--save_curves", action="store_true",
                    help="Save per-sample deletion curves")

args = parser.parse_args()


# ============================================================
# Paths & setup
# ============================================================

run_folder = Path(args.run_folder)
dataset_name = run_folder.parent.name
model_name = run_folder.name.split("_", 1)[0]

path_run = Path(args.run_dir) / run_folder
results_path = Path(args.output_dir) / "results" / run_folder
results_root = results_path / "evaluation" / "deletion"
saliency_root = results_path / args.xai_method

results_root.mkdir(parents=True, exist_ok=True)

assert saliency_root.exists(), f"Saliency folder not found: {saliency_root}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


# ============================================================
# Load model
# ============================================================

model = DinoV2ClassifierSlice.load_best_checkpoint(path_run)
model.to(device).eval()

patch_size = model.encoder.patch_embed.patch_size
if isinstance(patch_size, tuple):
    patch_size = patch_size[0]


# ============================================================
# Load dataset
# ============================================================

if dataset_name == "ODELIA":
    dataset = ODELIA_Dataset3D(split="test")
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")



# ============================================================
# Collect saliency files
# ============================================================

saliency_files = sorted(
    saliency_root.glob("class_*/pt/*_importance.pt")
)

if args.max_samples != -1:
    saliency_files = saliency_files[: args.max_samples]

assert len(saliency_files) > 0, "No saliency files found"


# --------------------------------------------------
# Build UID → index map ONLY for needed samples
# --------------------------------------------------

uids_needed = {
    pt.stem.replace("_importance", "")
    for pt in saliency_files
}

uid_to_index = {}

for i in range(len(dataset)):
    sample = dataset[i]
    uid = str(sample["uid"])

    if uid in uids_needed:
        uid_to_index[uid] = i

    if len(uid_to_index) == len(uids_needed):
        break


# ============================================================
# Prepare output folders
# ============================================================


curve_root = results_root / "deletion_curves" / args.xai_method
if args.save_curves:
    curve_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# Deletion loop
# ============================================================

rows = []

pbar = tqdm(
    saliency_files,
    desc=f"Deletion ({args.xai_method})",
    ncols=100,
)

for pt_path in pbar:
    uid = pt_path.stem.replace("_importance", "")

    if uid not in uid_to_index:
        continue

    idx = uid_to_index[uid]
    sample = dataset[idx]

    batch = {
        "source": sample["source"].unsqueeze(0).to(device)
    }

    # ---------- Load saliency ----------
    saliency = load_saliency(pt_path, device=device)

    # ---------- Prediction ----------
    with torch.no_grad():
        logits = model(batch["source"])
        pred = logits.argmax(dim=1).item()

    # ---------- Deletion evaluation ----------
    percentages, confidences, auc_score = deletion_evaluation(
        model=model,
        batch=batch,
        saliency=saliency,
        target_class=pred,
        patch_size=patch_size,
        steps=args.steps,
        replacement=args.replacement,
    )

    rows.append({
        "UID": uid,
        "dataset": dataset_name,
        "model": model_name,
        "xai_method": args.xai_method,
        "predicted_class": pred,
        "deletion_auc": auc_score,
    })

    # ---------- Save curve (optional) ----------
    if args.save_curves:
        np.save(
            curve_root / f"{uid}_curve.npy",
            {
                "percentages": percentages,
                "confidences": confidences,
                "auc": auc_score,
            },
            allow_pickle=True
        )

    pbar.set_postfix(auc=f"{auc_score:.3f}")


# ============================================================
# Save results
# ============================================================

df = pd.DataFrame(rows)

# Per-sample AUC
per_sample_csv = results_root / f"deletion_{args.xai_method}.csv"
df.to_csv(per_sample_csv, index=False)

# Aggregated summary
summary = (
    df.groupby("xai_method")["deletion_auc"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

summary_csv = results_root / f"deletion_{args.xai_method}_summary.csv"
summary.to_csv(summary_csv, index=False)


# ============================================================
# Final output
# ============================================================

print("\nDeletion evaluation finished.")
print(f"Per-sample AUC saved to: {per_sample_csv}")
print(f"Summary saved to:        {summary_csv}")
print("\nSummary:")
print(summary)
