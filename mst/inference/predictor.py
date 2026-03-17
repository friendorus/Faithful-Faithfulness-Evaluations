from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from mst.data.datasets.dataset_3d_odelia import ODELIA_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.models.dino import DinoV2ClassifierSlice
from mst.models.resnet import ResNet, ResNetSliceTrans
import torch
from pathlib import Path
import platform
import os
from mst.utils.ignore_warning import suppress_mst_warnings
suppress_mst_warnings()


# --------------------------------------------------
# Import supported MST model architectures.


def get_model_class(name):
    if name == "ResNet":
        return ResNet
    elif name == "ResNetSliceTrans":
        return ResNetSliceTrans
    elif name == "DinoV2ClassifierSlice":
        return DinoV2ClassifierSlice
    else:
        raise ValueError(f"Unknown model: {name}")


# --------------------------------------------------
# Import supported MST dataset classes.


def get_dataset_class(name):
    if name == "DUKE":
        return DUKE_Dataset3D
    elif name == "LIDC":
        return LIDC_Dataset3D
    elif name == "MRNet":
        return MRNet_Dataset3D
    elif name == "ODELIA":
        return ODELIA_Dataset3D
    else:
        raise ValueError(f"Unknown dataset: {name}")


# --------------------------------------------------
# Model loading utility [Load Model from Checkpoint]
def load_model(model_name, checkpoint_path, device):

    ModelClass = get_model_class(model_name)

    # Case 1: user passed a checkpoint file
    if checkpoint_path.is_file():

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Lightning checkpoint
        if isinstance(checkpoint, dict) and "pytorch-lightning_version" in checkpoint:
            model = ModelClass.load_from_checkpoint(str(checkpoint_path))
        # Non-Lightning checkpoint
        else:
            model = ModelClass(
                in_ch=3,
                out_ch=3,
                dino='v3-vit',
                model_size= 'b'
            )
            # Lightning-like state_dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            # Training checkpoint (your case)
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            # Raw weights
            else:
                model.load_state_dict(checkpoint)

    # Case 2: user passed a run directory -> load best checkpoint
    elif checkpoint_path.is_dir():
        model = ModelClass.load_best_checkpoint(str(checkpoint_path))
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    model.to(device)

    model.eval()

    return model


# --------------------------------------------------
# Batch prediction utility [Predict Batch with Optional Test-Time Augmentation (TTA)]
# import torch


def predict_batch(model, batch, device, use_tta=False):

    source = batch["source"].to(device)
    mask = batch.get("src_key_padding_mask", None)
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device)

    def forward_pass(x, m):
        out = model(x, src_key_padding_mask=m)
        return torch.softmax(out, dim=-1)

    with torch.no_grad():
        pred = forward_pass(source, mask)

        if use_tta:
            flip_dims = [
                (2,), (3,), (4,),
                (2, 3), (2, 4), (3, 4),
                (2, 3, 4)
            ]

            for dims in tqdm(flip_dims, desc="TTA Flips", leave=False):
                flipped = torch.flip(source, dims)
                pred += forward_pass(flipped, mask)

            pred /= (1 + len(flip_dims))

    return pred


# --------------------------------------------------
# Inference runner utility [Run Inference on Dataset and Collect Results]


def run_inference(
    model,
    dataset_name,
    device,
    use_tta=False,
    batch_size=1,
    num_workers=8,
):

    DatasetClass = get_dataset_class(dataset_name)
    dataset = DatasetClass(split="test")

    num_workers = 0 if platform.system() == "Darwin" else 8

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    results = []

    for batch in tqdm(loader, desc="Running Inference", dynamic_ncols=True):

        target = batch["target"]
        uid = batch["uid"][0] if isinstance(
            batch["uid"], list) else str(batch["uid"].item())

        probs = predict_batch(model, batch, device, use_tta=use_tta).cpu()

        pred_class = torch.argmax(probs, dim=1)
        pred_conf = probs.max(dim=1).values

        for b in range(target.shape[0]):
            row = {
                "UID": uid,
                "GT": int(target[b].item()),
                "NN": int(pred_class[b].item()),
                "NN_conf": float(pred_conf[b].item()),
            }

            for c, p in enumerate(probs[b].tolist()):
                row[f"prob_{c}"] = float(p)

            results.append(row)

    return pd.DataFrame(results)
# --------------------------------------------------
