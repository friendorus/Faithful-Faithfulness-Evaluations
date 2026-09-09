import subprocess
import itertools
import sys

RUN_FOLDER = "DINOv3ViTB"
CHECKPOINT = "challenge_mstv3-vit_sch_CB_sub2_best.chkpt"  # specify the checkpoint to evaluate

BASELINES = [
    # "maximum-intensity",
    # "gaussian_blur",
    # "minimum-intensity",
    # "black-3",
    # "black-5",
    # "black-10",
    # "black-8",
    # "white-5",
    # "black-5"
    # "white-10",
    # "zero",
    # "mean",
    "attention_mask",
]

XAI_METHODS = [
    # "gradcam",
    # # "last_layer",
    # # "slice_weighted_rollout",
    # "hires_cam",
    # "grad_sam",
    # # "gradcam_no_relu",
    # # "hires_cam_no_relu",
    # "random",
    # # "nonclass_gradcam",
    # # "nonclass_hires_cam",
    "gmar_l1",
    "gmar_l2",
    "nonclass_gmar_l1",
    "nonclass_gmar_l2",
    # "grad_rollout",
    # "nonclass_grad_sam",
    # "nonclass_grad_rollout",

]

COMMON_ARGS = [
    "python", "scripts/run_perturbation_evaluation.py",
    "--run_folder", RUN_FOLDER,
    "--checkpoint_name", CHECKPOINT,
    "--dataset", "ODELIA",
    # "--mode", "all",
    "--mode", "insertion",
    "--steps", "20",
    "--save_curves",
]

STOP_ON_ERROR = False  # set True if you want it to abort on first crash


def run():
    total = len(BASELINES) * len(XAI_METHODS)
    counter = 1

    for baseline, xai in itertools.product(BASELINES, XAI_METHODS):
        cmd = COMMON_ARGS + [
            "--xai_method", xai,
            "--replacement", baseline
        ]

        print(f"\n[{counter}/{total}] Running: XAI={xai} | BASELINE={baseline}")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"FAILED: XAI={xai} BASELINE={baseline}")
            if STOP_ON_ERROR:
                sys.exit(result.returncode)

        counter += 1

    print("\nAll experiments finished.")


if __name__ == "__main__":
    run()