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
    baseline: str = "minimum-intensity",  # "minimum-intensity" | "attention_mask" | "black-3" | "black-5" | "black-10" | "white-5" | "white-10" | "zero" | "mean" | "zero_conf" | "gaussian" 
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
            How to method to replace patches or being initial image ["black-3" | "black-5" | "black-10" | "white-5" | "white-10" | "zero" | "mean" | "zero_conf" | "gaussian"]
            except "attention_mask" For "attention_mask" baseline, we will use the original source as the initial image and rely on the patch_mask to control which patches are considered by the model.
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
    # source = batch["source"].clone()    # [1, C, D, H, W]

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
    # 3 Initial image & replacement
    # --------------------------------------------------
    if baseline.startswith("black-"):
        # Extract numeric value after "black-"
        try:
            black_value = -float(baseline.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid baseline format: {baseline}. Expected 'black-<number>'.")
        repl = torch.full_like(source, black_value)

    elif baseline.startswith("white-"):
        # Extract numeric value after "white-"
        try:
            white_value = float(baseline.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid baseline format: {baseline}. Expected 'white-<number>'.")
        repl = torch.full_like(source, white_value)
    
    elif baseline == "minimum-intensity":
        min_value = source.min()
        repl = torch.full_like(source, min_value)

    elif baseline == "zero": #Zeroing out the patch (setting to zero) - for MRI, zeroing out is not blackening, but setting to zero value of MRI
        repl = torch.zeros_like(source)

    elif baseline == "mean":
        repl = source.mean() * torch.ones_like(source)

    elif baseline == "gaussian":
        repl = gaussian_blur_3d(source)
        assert repl.shape == source.shape  

    elif baseline == "zero_conf": #Using baseline patch that have zero confidence
        assert reference_source is not None, \
            "reference_source required for zero_conf replacement"
        repl = reference_source.clone()
    
    elif baseline == "attention_mask": # Using attention mask instead of replace patch with specific value, so no need to create repl tensor.
        repl = None # Model should handle this case internally by using patch_mask to ignore those patches, so no need to create a separate repl tensor.

    else:
        raise ValueError(f"Unknown replacement: {baseline}")
    

    if baseline == "attention_mask":
        # --------------------------------------------------
        # Attention mask (True = keep, False = remove)
        # --------------------------------------------------
        patch_mask = torch.ones(
            (D, H_p, W_p),
            device=device,
            dtype=torch.bool
        )
        # For attention mask baseline, we will use the original source as the initial image and rely on the patch_mask to control which patches are considered by the model.
        current = source.clone()  # Start with the original image

        if mode == "insertion": # attention mask + insertion
            patch_mask[:] = False # Start with all patches removed (False) and add them back in order
        else: # attention mask + deletion or attention mask + negative perturbation
            patch_mask[:] = True # Start with all patches present (True) and remove them in order

    else:
        if mode == "insertion":
            current = repl.clone() #using baseline value as Initial image
        else:
            current = source.clone() #using original input image as initial images


    raw_logits = []
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
                if baseline != "attention_mask":
                    current[:, :, d, h0:h1, w0:w1] = source[:, :, d, h0:h1, w0:w1] #source - keep current area with that patch
                else: # insertion + attention mask
                    patch_mask[d, h, w] = True # Mark patch as present
            else: # deletion or negative perturbation
                if baseline != "attention_mask":
                    current[:, :, d, h0:h1, w0:w1] = repl[:, :, d, h0:h1, w0:w1] #replace current area from patch to mask value
                else: # deletion + attention mask or negative perturbation + attention mask
                    patch_mask[d, h, w] = False # Mark patch as removed

        logits = model(current, 
                       patch_mask=patch_mask if baseline == "attention_mask" else None)         # logits of model from current perturbed input
        # prob = torch.softmax(logits, dim=1)[0, predicted_class] # convert logits into problability and select that prob to the class of choice.
        # confidences.append(prob.item()) #Count Prob of only that class as confidences
        raw_logits.append(logits.detach().cpu()) #Store raw logits for all classes for later normalization


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

    return percentages, raw_logits,confidences, confidences_normalized, auc_score


import torch
import torch.nn.functional as F


@torch.no_grad()
def gaussian_blur_3d(
    x: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 2.0,
    ) -> torch.Tensor:
    """
    Shape-preserving Gaussian blur for 3D volumes.

    Input:
        x: Tensor of shape [C, D, H, W] or [B, C, D, H, W]

    Output:
        Tensor with EXACT same shape as x
    """

    assert x.dim() in (4, 5), f"Expected 4D or 5D tensor, got {x.shape}"

    # Track original shape
    has_batch = (x.dim() == 5)

    if not has_batch:
        x = x.unsqueeze(0)  # [1, C, D, H, W]

    B, C, D, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # 1D Gaussian kernel
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()

    # Separable kernels
    kz = kernel_1d.view(1, 1, kernel_size, 1, 1)
    ky = kernel_1d.view(1, 1, 1, kernel_size, 1)
    kx = kernel_1d.view(1, 1, 1, 1, kernel_size)

    # Expand for grouped convolution (per channel)
    kz = kz.repeat(C, 1, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1, 1)
    kx = kx.repeat(C, 1, 1, 1, 1)

    padding = kernel_size // 2

    # Depth
    x = F.conv3d(
        x,
        kz,
        padding=(padding, 0, 0),
        groups=C,
    )

    # Height
    x = F.conv3d(
        x,
        ky,
        padding=(0, padding, 0),
        groups=C,
    )

    # Width
    x = F.conv3d(
        x,
        kx,
        padding=(0, 0, padding),
        groups=C,
    )

    if not has_batch:
        x = x.squeeze(0)

    return x