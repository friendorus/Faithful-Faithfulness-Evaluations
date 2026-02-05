import torch
import torch.nn.functional as F
from sklearn.metrics import auc


@torch.no_grad()
def perturbation_evaluation(
    model,
    batch,
    saliency,                 # torch.Tensor [D, H, W]
    predicted_class: int,
    patch_size: int,
    steps: int,                 # How often that process will be evaluated" - 20 mean every 5%
    mode: str,                # "deletion" | "insertion" | "negative"
    baseline: str = "black-5",  # "black-5" | "black-10" | "zero" | "mean" | "zero_conf"
    reference_source: torch.Tensor | None = None,
):
    """
    Generic patch-level perturbation evaluation.

        Purturbation evaluation ***(patch-level).

        Parameters
        ----------
        model : torch.nn.Module
            Trained MST model
        batch : dict
            Must contain:
                batch["source"] : [1, C, D, H, W]
        saliency : torch.Tensor
            Saliency map [D, H, W]
        predicted_class : int
            Class that was predicted by the model
        patch_size : int
            ViT patch size (e.g. 14 or 16)
        steps : int
            Number of add/remove steps - standard is 20
        mode : str
            Mode to use for evaluation [Deletion, Insertion or Negative (Perturbation)]
        baseline : str
            How to method to replace patches or being initial image ["black with -5" | "black with -10" | "zero" | "mean" | "zero_conf"]
        reference_source : torch.Tensor
            if using zero cofidence, require patchs that want to replace

        Returns
        -------
        percentages : list[float]
            Fraction of patches deleted
        confidences : list[float]
            Model confidence for target_class
        auc_score : float
            Area under deletion curve
        confidence calculation mode : str
            Mode to confirm what use to calculate confidence

    """

    device = batch["source"].device     #ensure all tensors are on the same device (CPU or GPU)
    source = batch["source"].clone()    # [1, C, D, H, W] # clone () to prevents modifying original input

    B, C, D, H, W = source.shape        #[Batch=1, Channels, Depth, Height, Width]
    assert B == 1, "Expect Batch size = 1"
    source = batch["source"].clone()    # [1, C, D, H, W]

    # --------------------------------------------------
    # 1. Saliency → patch grid [Convert Voxel-level saliency into patch-level saliency]
    # --------------------------------------------------
    H_p = H // patch_size
    W_p = W // patch_size

    sal_patch = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),     # Downsample saliency to patch resolution
        size=(D, H_p, W_p),                     # [D, H, W] → [1, 1, D, H, W]
        mode="trilinear",                       # Trilinear interpolation (for 3D data)
        align_corners=False
    )[0, 0] # [D, H_p, W_p]

    flat_sal = sal_patch.flatten()              # Change shape [D, H_p, W_p] → [D × H_p × W_p]

    # --------------------------------------------------
    # 2. Patch ordering
    # --------------------------------------------------
    if mode in ["deletion", "insertion"]:
        order = torch.argsort(flat_sal, descending=True) # Rank from highest score to lowest score
    elif mode == "negative": #negative perturbation test
        order = torch.argsort(flat_sal, descending=False) # Rank from lowest score to highest score
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_patches = flat_sal.numel()            # count totla number of patches D × H_p × W_p

    # --------------------------------------------------
    # 3. Initial image & replacement
    # --------------------------------------------------
    if baseline == "black-5":  # REAL blackening for MRI
        black_value = -5.0
        repl = torch.full_like(source, black_value)
    elif baseline == "black-10":  # REAL blackening for MRI
        black_value = -10.0
        repl = torch.full_like(source, black_value)
    elif baseline == "zero": #Zeroing out the patch (setting to zero) - for MRI, zeroing out is not blackening, but setting to zero value of MRI
        repl = torch.zeros_like(source)
    elif baseline == "mean":
        repl = source.mean() * torch.ones_like(source)
    elif baseline == "zero_conf": #Insert baseline patch that have zero conference
        assert reference_source is not None, \
            "reference_source required for zero_conf replacement"
        repl = reference_source.clone()
    else:
        raise ValueError(f"Unknown replacement: {baseline}")

    if mode == "insertion":
        current = repl.clone() #using baseline value as Initial image
    else:
        current = source.clone() #using original input image as initial images

    confidences = []
    percentages = []

    # --------------------------------------------------
    # 4. Perturbation loop
    # --------------------------------------------------
    prev_k = 0
    for step in range(steps + 1):
        k = int(step / steps * total_patches)
        idxs = order[prev_k:k]

        for idx in idxs:
            d = idx // (H_p * W_p)
            hw = idx % (H_p * W_p)
            h = hw // W_p
            w = hw % W_p

            h0, h1 = h * patch_size, (h + 1) * patch_size
            w0, w1 = w * patch_size, (w + 1) * patch_size

            if mode == "insertion":
                current[:, :, d, h0:h1, w0:w1] = source[:, :, d, h0:h1, w0:w1] #source - keep current area with that patch
            else:
                current[:, :, d, h0:h1, w0:w1] = repl[:, :, d, h0:h1, w0:w1] #replace current area from patch to mask value

        logits = model(current)         # logits of model from current perturbed input
        # prob = torch.softmax(logits, dim=1)[0, predicted_class] # convert logits into problability and select that prob to the class of choice.
        # confidences.append(prob.item()) #Count Prob of only that class as confidences

        prob = torch.softmax(logits, dim=1)[0]  # Get probabilities for all classes
        confidences.append(prob.detach().cpu())  # Store all class probabilities

        percentages.append(step / steps)

        prev_k = k  # Update previous k for next iteration
    # --------------------------------------------------
    # 5. Normalization to 0,1 (Realative Confidence)
    # --------------------------------------------------

    confidences = torch.stack(confidences)  # Convert list to tensor [steps+1, num_classes]
    # Normalize all classes independently
    conf_min = confidences.min(dim=0, keepdim=True)[0]
    conf_max = confidences.max(dim=0, keepdim=True)[0]

    confidences_normalized = (
        (confidences - conf_min) /
        (conf_max - conf_min + 1e-8)
    )

    # AUC on normalized predicted class curve
    pred_curve = confidences_normalized[:, predicted_class]
    auc_score = auc(percentages, pred_curve.tolist())

    return percentages, confidences, confidences_normalized, auc_score

