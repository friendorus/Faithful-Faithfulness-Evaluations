import subprocess
import itertools
import sys

RUN_FOLDER = "ODELIA/DinoV2ClassifierSlice_Final"

BASELINES = [
    "minimum-intensity",
    "black-3",
    "black-5",
    "black-10",
    "white-5",
    "white-10",
    "zero",
    "mean",
    "gaussian_blur",
    "attention_mask",
]

XAI_METHODS = [
    "gradcam",
    "last_layer",
    #"rollout",
    "slice_weighted_rollout",
]

COMMON_ARGS = [
    "python", "scripts/run_perturbation_evaluation.py",
    "--run_folder", RUN_FOLDER,
    "--mode", "all",
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
            "--baseline", baseline
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