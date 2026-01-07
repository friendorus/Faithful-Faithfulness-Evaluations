import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc


@torch.no_grad()
def deletion_evaluation(
    model,
    batch,
    saliency,                 # torch.Tensor [D, H, W]
    target_class: int,
    patch_size: int,
    steps: int = 20,
    replacement: str = "zero",   # "zero" | "mean"
):
    """
    Deletion faithfulness evaluation (patch-level).

    Parameters
    ----------
    model : torch.nn.Module
        Trained MST model
    batch : dict
        Must contain:
            batch["source"] : [1, C, D, H, W]
    saliency : torch.Tensor
        Saliency map [D, H, W]
    target_class : int
        Class index to track confidence
    patch_size : int
        ViT patch size (e.g. 14 or 16)
    steps : int
        Number of deletion steps
    replacement : str
        How to replace deleted patches

    Returns
    -------
    percentages : list[float]
        Fraction of patches deleted
    confidences : list[float]
        Model confidence for target_class
    auc_score : float
        Area under deletion curve
    """

    device = batch["source"].device
    source = batch["source"].clone()  # [1, C, D, H, W]

    B, C, D, H, W = source.shape
    assert B == 1, "Deletion expects batch size = 1"

    # --------------------------------------------------
    # 1. Downsample saliency to PATCH GRID (2D per slice)
    # --------------------------------------------------
    H_p = H // patch_size
    W_p = W // patch_size

    sal_patch = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),  # [1,1,D,H,W]
        size=(D, H_p, W_p),
        mode="trilinear",
        align_corners=False
    )[0, 0]  # [D, H_p, W_p]

    # --------------------------------------------------
    # 2. Rank patches by importance
    # --------------------------------------------------
    flat_sal = sal_patch.flatten()          # [D*H_p*W_p]
    order = torch.argsort(flat_sal, descending=True)

    total_patches = flat_sal.numel()

    # --------------------------------------------------
    # 3. Replacement tensor
    # --------------------------------------------------
    if replacement == "zero":
        repl = torch.zeros_like(source)
    elif replacement == "mean":
        repl = source.mean() * torch.ones_like(source)
    else:
        raise ValueError(f"Unknown replacement: {replacement}")

    # --------------------------------------------------
    # 4. Deletion loop
    # --------------------------------------------------
    confidences = []
    percentages = []

    for step in range(steps + 1):
        k = int(step / steps * total_patches)
        mask_idx = order[:k]

        masked = source.clone()

        for idx in mask_idx:
            d = idx // (H_p * W_p)
            hw = idx % (H_p * W_p)
            h = hw // W_p
            w = hw % W_p

            h0, h1 = h * patch_size, (h + 1) * patch_size
            w0, w1 = w * patch_size, (w + 1) * patch_size

            masked[:, :, d, h0:h1, w0:w1] = repl[:, :, d, h0:h1, w0:w1]

        logits = model(masked)
        prob = torch.softmax(logits, dim=1)[0, target_class]

        confidences.append(prob.item())
        percentages.append(step / steps)

    # --------------------------------------------------
    # 5. AUC (lower = better faithfulness)
    # --------------------------------------------------
    auc_score = auc(percentages, confidences)

    return percentages, confidences, auc_score
