import argparse
import torch
from pathlib import Path

from mst.inference.predictor import load_model, run_inference
from mst.inference.evaluation import evaluate_and_plot  # optional separate file        


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = Path(args.run_folder).name.split("_", 1)[0]

    model = load_model(
        model_name=model_name,
        checkpoint_path=Path(args.run_folder),
        device=device,
    )

    df = run_inference(
        model=model,
        dataset_name=args.dataset,
        device=device,
        use_tta=args.use_tta,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)

    evaluate_and_plot(df, str(output_dir))

    print("Saved results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--run_folder", type=str, required=True, help="Folder name of the run")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save predictions")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    args = parser.parse_args()
    main(args)
