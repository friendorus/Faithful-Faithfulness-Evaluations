import torch
import torch.nn.functional as F
from sklearn.metrics import auc


@torch.no_grad()
def insertion_evaluation(
    model,
    batch,
    saliency,                 # torch.Tensor [D, H, W]
    target_class: int,
    patch_size: int,
    steps: int = 20,
    baseline: str = "zero",    # "zero" | "mean" | "zero_conf"
    reference_source: torch.Tensor | None = None,
):
    if baseline == "zero_conf":
        assert reference_source is not None, \
            "reference_source required for zero_conf baseline"

    device = batch["source"].device
    source = batch["source"].clone()  # [1, C, D, H, W]

    B, C, D, H, W = source.shape
    assert B == 1, "Insertion expects batch size = 1"

    # --------------------------------------------------
    # 1. Downsample saliency to PATCH GRID
    # --------------------------------------------------
    H_p = H // patch_size
    W_p = W // patch_size

    sal_patch = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),
        size=(D, H_p, W_p),
        mode="trilinear",
        align_corners=False
    )[0, 0]  # [D, H_p, W_p]

    # --------------------------------------------------
    # 2. Rank patches by importance (descending)
    # --------------------------------------------------
    flat_sal = sal_patch.flatten()
    order = torch.argsort(flat_sal, descending=True)

    total_patches = flat_sal.numel()

    # --------------------------------------------------
    # 3. Initialize baseline image
    # --------------------------------------------------
    if baseline == "zero":
        current = torch.zeros_like(source)
    elif baseline == "mean":
        current = source.mean() * torch.ones_like(source)
    elif baseline == "zero_conf":
        current = reference_source.clone()
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    confidences = []
    percentages = []

    # --------------------------------------------------
    # 4. Insertion loop
    # --------------------------------------------------
    for step in range(steps + 1):
        k = int(step / steps * total_patches)
        insert_idx = order[:k]

        for idx in insert_idx:
            d = idx // (H_p * W_p)
            hw = idx % (H_p * W_p)
            h = hw // W_p
            w = hw % W_p

            h0, h1 = h * patch_size, (h + 1) * patch_size
            w0, w1 = w * patch_size, (w + 1) * patch_size

            current[:, :, d, h0:h1, w0:w1] = source[:, :, d, h0:h1, w0:w1]

        logits = model(current)
        prob = torch.softmax(logits, dim=1)[0, target_class]

        confidences.append(prob.item())
        percentages.append(step / steps)

    # --------------------------------------------------
    # 5. AUC (higher = better faithfulness)
    # --------------------------------------------------
    auc_score = auc(percentages, confidences)

    return percentages, confidences, auc_score
