import torch
import torch.nn.functional as F
from sklearn.metrics import auc


@torch.no_grad()
def perturbation_evaluation(
    model,
    batch,
    saliency,                 # torch.Tensor [D, H, W]
    target_class: int,
    predicted_class: int,
    patch_size: int,
    steps: int,                 # How often that process will be evaluated" - 20 mean every 5%
    mode: str,                # "deletion" | "insertion" | "negative"
    confidence_calculation_mode: str, # "GroundTruth" | "Predicted"
    baseline: str = "zero",  # "zero" | "mean" | "zero_conf"
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
        target_class : int
            Ground Truth Class index to track confidence if want to track on GT
        predicted_class : int
            Class that was predicted by the model
        patch_size : int
            ViT patch size (e.g. 14 or 16)
        steps : int
            Number of add/remove steps - standard is 20
        mode : str
            Mode to use for evaluation [Deletion, Insertion or Negative (Perturbation)]
        confidence_calculation_mode: str
            What is the mode to use for confidence calculation
        baseline : str
            How to method to replace patches or being initial image
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

    device = batch["source"].device
    source = batch["source"].clone()  # [1, C, D, H, W]

    B, C, D, H, W = source.shape
    assert B == 1, "Expect Batch size = 1"

    # --------------------------------------------------
    # 1. Saliency → patch grid
    # --------------------------------------------------
    H_p = H // patch_size
    W_p = W // patch_size

    sal_patch = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),
        size=(D, H_p, W_p),
        mode="trilinear",
        align_corners=False
    )[0, 0] # [D, H_p, W_p]

    flat_sal = sal_patch.flatten()

    # --------------------------------------------------
    # 2. Patch ordering
    # --------------------------------------------------
    if mode in ["deletion", "insertion"]:
        order = torch.argsort(flat_sal, descending=True) # Rank from highest score to lowest score
    elif mode == "negative": #negative perturbation test
        order = torch.argsort(flat_sal, descending=False) # Rank from lowest score to highest score
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_patches = flat_sal.numel()

    # --------------------------------------------------
    # 3. Initial image & replacement
    # --------------------------------------------------
    if baseline == "zero": #Blackening
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
    for step in range(steps + 1):
        k = int(step / steps * total_patches)
        idxs = order[:k]

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

        logits = model(current)
        if confidence_calculation_mode == "GroundTruth":
            prob = torch.softmax(logits, dim=1)[0, target_class]
        elif confidence_calculation_mode == "Predicted":
            prob = torch.softmax(logits, dim=1)[0, predicted_class]
        else:
            raise ValueError(f"Unknown what is the class to calculate confidence")

        confidences.append(prob.item())
        percentages.append(step / steps)

    auc_score = auc(percentages, confidences)

    return percentages, confidences, auc_score, confidence_calculation_mode
