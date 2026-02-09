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

from mst_xai.utils.load_saliency import load_saliency


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(description="Unified perturbation-based faithfulness evaluation")
parser.add_argument("--run_dir", default="./runs", type=str)
parser.add_argument("--run_folder", required=True, type=str)
parser.add_argument("--output_dir", default="./", type=str)
parser.add_argument("--mode", required=True, choices=["deletion", "insertion", "negative", "all"], help="Perturbation mode to run",)
parser.add_argument("--xai_method", required=True, choices=["attention", "attention_rollout"], help="Which saliency 'folder' to evaluate",)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all available saliency files")
parser.add_argument("--baseline", default="minimum-intensity", choices=["minimum-intensity", "black-3", "black-5", "black-10", 
                                                                        "white-5", "white-10", "zero", "mean", "zero_conf", "gaussian", "attention_mask"], 
                                                                        help="Baseline for perturbation. Use 'attention_mask' to mask patches in attention instead of replacing pixel values.",)
parser.add_argument("--save_curves", action="store_true")

args = parser.parse_args()


# ============================================================
# Paths & setup
# ============================================================

run_folder = Path(args.run_folder)
dataset_name = run_folder.parent.name
model_name = run_folder.name.split("_", 1)[0]

path_run = Path(args.run_dir) / run_folder
results_path = Path(args.output_dir) / "results" / run_folder
saliency_root = results_path / "saliency_results" / args.xai_method

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
# UID → dataset index map
# --------------------------------------------------

uids_needed = {
    pt.stem.replace("_importance", "")
    for pt in saliency_files
}

uid_to_index = {}

for i in range(len(dataset)):
    uid = str(dataset[i]["uid"])
    if uid in uids_needed:
        uid_to_index[uid] = i
    if len(uid_to_index) == len(uids_needed):
        break

# ============================================================
# Determine modes to run
# ============================================================

if args.mode == "all":
    modes = ["deletion", "insertion", "negative"]
else:
    modes = [args.mode]

# ============================================================
# Run evaluations
# ============================================================

for mode in modes:
    if mode == "deletion":
        from mst_xai.evaluation_methods.deletion import deletion_evaluation
        eval_fn = deletion_evaluation

    elif mode == "insertion":
        from mst_xai.evaluation_methods.insertion import insertion_evaluation
        eval_fn = insertion_evaluation

    elif mode == "negative":
        from mst_xai.evaluation_methods.negative_perturbation import negative_perturbation_evaluation
        eval_fn = negative_perturbation_evaluation


    print(f"\n=== Running {mode.upper()} evaluation ===")

    results_root = results_path / "evaluation_results" / mode
    results_root.mkdir(parents=True, exist_ok=True)

    curve_root = results_root / f"{mode}_curves" / args.xai_method / args.baseline

    if args.save_curves:
        curve_root.mkdir(parents=True, exist_ok=True)

    rows = []
    intensity = []

    pbar = tqdm(
        saliency_files,
        desc=f"{mode.capitalize()}",
        ncols=100,
    )

    for pt_path in pbar:
        uid = pt_path.stem.replace("_importance", "")

        if uid not in uid_to_index:
            continue

        sample = dataset[uid_to_index[uid]]

        batch = {
            "source": sample["source"].unsqueeze(0).to(device)
        }

        # ---------- Load saliency ----------
        saliency = load_saliency(pt_path, device=device)

        # ----------------------------------------------
        # Model's original prediction (reference class)
        # ----------------------------------------------
        with torch.no_grad():
            logits = model(batch["source"])
            predicted_class = logits.argmax(dim=1).item()

        # ----------------------------------------------
        # Perturbation evaluation
        # ----------------------------------------------
        percentages, raw_logits, confidences, confidences_normalized, auc_score = eval_fn(
            model=model,
            batch=batch,
            saliency=saliency,
            predicted_class=predicted_class,
            patch_size=patch_size,
            steps=args.steps,
            baseline=args.baseline,
        )

        predicted_confidences = confidences[:, predicted_class]
        predicted_confidences_normalized = confidences_normalized[:, predicted_class]

            # DEBUG: check post-preprocessing intensity range (print once)

        print ("Post-preprocessing intensity stats:",
            "min =", batch["source"].min().item(),
            "mean =", batch["source"].mean().item(),
            "max =", batch["source"].max().item())
        intensity.append([uid, batch["source"].min().item(), batch["source"].mean().item(), batch["source"].max().item()])
        # print ("----------------------------------------------")
            
        # print(f"Predicted class is {predicted_class}")
        # print(f"Progression percentage is {percentages}")
        # print(f"Confidence score is {confidences}")
        # print(f"Normalized Confidence score is {confidences_normalized}")
        # print(f"Predicted class confidence: {predicted_confidences}")
        # print(f"Normalized predicted class confidence: {predicted_confidences_normalized}")


        rows.append({
            "UID": uid,
            "dataset": dataset_name,
            "model": model_name,
            "xai_method": args.xai_method,
            "mode": mode,
            "predicted_class": predicted_class,
            "auc": auc_score,
            "baseline": args.baseline,
        })

        curve_dict =  {
                    "percentages": np.array(percentages),
                    "raw_logits": torch.stack(raw_logits).cpu().numpy(),
                    "confidences": confidences.cpu().numpy(),
                    "normalized_confidences" : confidences_normalized.cpu().numpy(),
                    "predicted_class": predicted_class,
                    "predicted_confidences": predicted_confidences.cpu().numpy(),
                    "predicted_normalized_confidences": predicted_confidences_normalized.cpu().numpy(),
                    "auc": auc_score,
                    "baseline": args.baseline,
                }

        if args.save_curves:
            np.save(
                curve_root / f"{uid}_curve_{args.baseline}.npy",
                curve_dict,
                allow_pickle=True,
            )

        pbar.set_postfix(auc=f"{auc_score:.3f}")

    # ========================================================
    # Save results
    # ========================================================

    df = pd.DataFrame(rows)

    per_sample_csv = results_root / f"{mode}_{args.xai_method}_{args.baseline}.csv"
    df.to_csv(per_sample_csv, index=False)

    summary = (
        df.groupby("xai_method")["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary_csv = results_root / f"{mode}_{args.xai_method}_{args.baseline}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    df_intensity = pd.DataFrame(intensity, columns=["UID", "min", "mean", "max"])
    intensity_csv = results_root / f"{mode}_{args.xai_method}_{args.baseline}_intensity_stats.csv"
    df_intensity.to_csv(intensity_csv, index=False)

    print(f"\n{mode.capitalize()} finished.")
    print(f"Per-sample CSV: {per_sample_csv}")
    print(f"Summary CSV:    {summary_csv}")
    print(f"Intensity stats CSV: {intensity_csv}")
    print(summary)


print("\nAll requested evaluations completed.")
