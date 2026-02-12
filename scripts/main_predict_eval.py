import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x")
warnings.filterwarnings("ignore", message=".*_ARRAY_API not found.*")



import argparse
import torch
from pathlib import Path

from mst.inference.predictor import load_model, run_inference
from mst.inference.evaluation import evaluate_and_plot  # optional separate file        


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_folder = Path(args.run_folder)
    model_name = run_folder.name.split("_", 1)[0]
    path_run = Path(args.run_dir) / run_folder
    results_path = Path(args.output_dir) / "results" / run_folder
    results_path.mkdir(parents=True, exist_ok=True)




    

    model = load_model(
        model_name=model_name,
        checkpoint_path=path_run,
        device=device,
    )

    df = run_inference(
        model=model,
        dataset_name=args.dataset,
        device=device,
        use_tta=args.use_tta,
    )

    output_dir = results_path / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)

    evaluate_and_plot(df, str(output_dir))

    print("Saved results.csv and evaluation plots to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--run_dir", default="./runs", type=str)
    parser.add_argument("--run_folder", type=str, required=True, help="Folder name of the run")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save predictions")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    args = parser.parse_args()
    main(args)
