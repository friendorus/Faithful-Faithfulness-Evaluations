import torch
import torch.nn.functional as F
from sklearn.metrics import auc


@torch.no_grad()
def perturbation_evaluation(
    model,
    batch,
    saliency,   # torch.Tensor [D, H, W] ([32, 224, 224])
    predicted_class: int,
    patch_size: int,
    steps: int, # How often that process will be evaluated" - 20 mean every 5%
    mode: str,  # "deletion" | "insertion" | "negative"
    replacement: str = "minimum-intensity",  # "minimum-intensity" | "attention_mask" | "black-3" | "black-5" | "black-10" | "white-5" | "white-10" | "zero" | "mean" | "zero_conf" | "gaussian_blur" 
    reference_source: torch.Tensor | None = None,
    return_images=False,
    slice_index=None # for specific slice visualization
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
        replacement : str
            How to method to replace patches or being initial image ["black-3" | "black-5" | "black-10" | "white-5" | "white-10" | "zero" | "mean" | "zero_conf" | "gaussian_blur"]
            except "attention_mask" For "attention_mask" replacement, we will use the original source as the initial image and rely on the patch_mask to control which patches are considered by the model.
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
    source = batch["source"].clone()    # clone () to prevents modifying original input

    B, C, D, H, W = source.shape        #[Batch=1, Channels, Depth, Height, Width]
    assert B == 1, "Expect Batch size = 1"

    num_special = 4 # 4 storage_tokens in DINOv3 (not include CLS token)

    def build_attn_mask_from_patch_mask(patch_mask, num_special):
        """
        Build attention mask that have size to insert into attention calculation
        Output shape: [D, 1, N, N] where N = 1 (CLS) + num_special + number of patch tokens
        """
        D, H_p, W_p = patch_mask.shape # [D, 14, 14]
        device = patch_mask.device 
        flat_mask = patch_mask.view(D, -1)  # [D, 196]


        offset = 1 + num_special # 1 = CLS
        N = offset + flat_mask.shape[1] # Total tokens = CLS + extra tokens + patch tokens

        attn_mask = torch.zeros(D, 1, N, N, device=device) # create zero attention mask [D, 1, N, N]

        for d in range(D):
            # Identify which patch tokens are masked 
            # In patch_mask, True means patch is present, False means patch is removed.
            # For attention mask, we want to set attention to/from masked patches to -inf
            # So we find the indices of the False patches in patch_mask and turn to True then set those to -inf in the attention mask.
            masked_tokens = (~flat_mask[d]).nonzero(as_tuple=True)[0] + offset

            # inf means no attention
            attn_mask[d, :, :, masked_tokens] = -1e9
            # attn_mask[d, :, masked_tokens, :] = -1e9
        return attn_mask
    
    # --------------------------------------------------
    # 1. Convert Saliency → patch grid 
    # --------------------------------------------------
    # For DINOv3 patch size = 16 x 16, H_p = 14, W_p = 14
    # For DINOv2 patch size = 14 x 14, H_p = 16, W_p = 16
    H_p = H // patch_size
    W_p = W // patch_size

    # saliency shape = [32, 224, 224] → [32, 14, 14] in DINOv3 case
    sal_patch = F.interpolate(                  # Downsample saliency to patch resolution
        saliency.unsqueeze(0).unsqueeze(0),     # [D, H, W] → [1, 1, D, H, W]
        size=(D, H_p, W_p),                     # [1, 1, D, H, W] → [1, 1, D, H_p, W_p]
        mode="trilinear",                       # Trilinear interpolation (for 3D data)
        align_corners=False
    )[0, 0]                                     # [1, 1, D, H_p, W_p] → [D, H_p, W_p]

    flat_sal = sal_patch.flatten()              # Change shape [D, H_p, W_p] → [D × H_p × W_p]

    # --------------------------------------------------
    # 2. Patch ordering
    # --------------------------------------------------
    if mode in ["deletion", "insertion"]:
        order = torch.argsort(flat_sal, descending=True) # Rank from highest to lowest 
    elif mode == "negative": #negative perturbation test
        order = torch.argsort(flat_sal, descending=False) # Rank from lowest to highest
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # count total number of patches D × H_p × W_p
    total_patches = flat_sal.numel() # 32 x 14 x 14 = 6272 patches in DINOv3 case

    # --------------------------------------------------
    # 3 Initial image & replacement
    # --------------------------------------------------

    # 3.1 Define replacement patch based on replacement strategy
    # "repl" will be the samse size as source and will take some spot to replace
    if replacement.startswith("black-"):
        # Extract numeric value after "black-"
        try:
            black_value = -float(replacement.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid replacement format: {replacement}. Expected 'black-<number>'.")
        repl = torch.full_like(source, black_value)

    elif replacement.startswith("white-"):
        # Extract numeric value after "white-"
        try:
            white_value = float(replacement.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid replacement format: {replacement}. Expected 'white-<number>'.")
        repl = torch.full_like(source, white_value)
    
    elif replacement == "minimum-intensity":
        min_value = source.min()
        repl = torch.full_like(source, min_value)

    elif replacement == "zero": 
        # Zeroing out the patch (setting to zero)
        # for MRI, zeroing out is not blackening, but setting to zero value of MRI
        repl = torch.zeros_like(source)

    elif replacement == "mean":
        # mean intensity of the whole image
        repl = source.mean() * torch.ones_like(source)

    elif replacement == "gaussian_blur": 
        # apply gaussian blur to the whole image (need to be at patch level, not image level)
        repl = gaussian_blur_patchwise(source, 
                                       patch_size=patch_size, 
                                       kernel_size=9, 
                                       sigma=2.0)

    elif replacement == "zero_conf": 
        assert reference_source is not None, \
            "reference_source required for zero_conf replacement"
        # replace whole image with any reference  # need to make it as same size as source
        repl = reference_source.clone()
   # For attenttion mask will not replace with any patch (keep original images)
    elif replacement == "attention_mask": 
        repl = None 

    else:
        raise ValueError(f"Unknown replacement: {replacement}")


    # 3.2 Define INITIAL image based on mode and replacement strategy

    # 3.2.1 For "Attention mask" replacement
    if replacement == "attention_mask":
        current = source.clone()  # Always start with the original image

        # Define dummy patch mask size [D, H_p, W_p]
        # to control which patches are visible or removed
        # True -> patch is visible, False -> patch is masked/removed
        patch_mask = torch.ones(
            (D, H_p, W_p),
            device=device,
            dtype=torch.bool) 

        if mode == "insertion": # attention mask + insertion
            patch_mask[:] = False # Start with all patches removed 
        else: 
            patch_mask[:] = True # Start with all patches present (True) 
    
    # 3.2.2 For non-attention mask replacement
    else:
        if mode == "insertion":
            #using replacement value as Initial image
            #For example, black-5, using all -5 intensity as inital image
            current = repl.clone() 
        else:
            current = source.clone() #using original input image as initial images


    

    # --------------------------------------------------
    # 4. Perturbation loop / Replacement process
    # --------------------------------------------------
    raw_logits = []
    confidences = [] # in this mean "probabilities" on each class
    percentages = []
    saved_images = [] if return_images else None

    prev_k = 0
    for step in range(steps + 1):
        k = int(step / steps * total_patches)
        idxs = order[prev_k:k]

        for idx in idxs:
            # convert index to localtion
            d = idx // (H_p * W_p)
            hw = idx % (H_p * W_p)
            h = hw // W_p
            w = hw % W_p

            h0, h1 = h * patch_size, (h + 1) * patch_size
            w0, w1 = w * patch_size, (w + 1) * patch_size

            # insertion
            if mode == "insertion":
                # Non-attention mask
                if replacement != "attention_mask":
                    # Replace current patch with ORIGINAL patch at same location
                    current[:, :, d, h0:h1, w0:w1] = source[:, :, d, h0:h1, w0:w1]
                # Attention mask 
                else: 
                    # mark patch that location as True to present in attention calculation
                    patch_mask[d, h, w] = True 
            
            # deletion or negative perturbation
            else: 
                if replacement != "attention_mask":
                    # replace current patch with replacement patch at same location
                    current[:, :, d, h0:h1, w0:w1] = repl[:, :, d, h0:h1, w0:w1] 
                else:
                    # mark patch that location as Flase to remove in attention calculation
                    patch_mask[d, h, w] = False 
        
        if replacement == "attention_mask":
            attn_mask = build_attn_mask_from_patch_mask(
                patch_mask,
                num_special=num_special)
            attn_mask = attn_mask.repeat(B, 1, 1, 1) # handle incase B != 1
        else:
            attn_mask = None

        if return_images: 
            img_slice = current[0,0, slice_index].detach().cpu().clone()
            saved_images.append(img_slice)

    
        logits = model(
            current,
            attn_mask=attn_mask)
        
        # store
        raw_logits.append(logits.detach().cpu()) #Store raw logits for all classes for later normalization

        prob = torch.softmax(logits, dim=-1)[0]  # Get probabilities for all classes
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

    if return_images:
        return percentages, raw_logits, confidences, confidences_normalized, auc_score, saved_images
    else:   
        return percentages, raw_logits, confidences, confidences_normalized, auc_score


import torch
import torch.nn.functional as F



def gaussian_blur_patchwise(
    x: torch.Tensor,
    patch_size: int,
    kernel_size: int = 9,
    sigma: float = 2.0,) -> torch.Tensor:

    assert x.dim() == 5, f"Expected [B,C,D,H,W], got {x.shape}"

    B, C, D, H, W = x.shape

    assert H % patch_size == 0
    assert W % patch_size == 0

    H_p = H // patch_size
    W_p = W // patch_size

    # --- reshape into patches ---
    x = x.view(B, C, D,
               H_p, patch_size,
               W_p, patch_size)

    x = x.permute(0, 2, 3, 5, 1, 4, 6)
    # [B, D, H_p, W_p, C, p, p]

    x = x.reshape(-1, C, 1, patch_size, patch_size)
    # each patch = independent sample

    # --- gaussian kernel ---
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords -= kernel_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    ky = kernel.view(1, 1, 1, kernel_size, 1).repeat(C, 1, 1, 1, 1)
    kx = kernel.view(1, 1, 1, 1, kernel_size).repeat(C, 1, 1, 1, 1)

    padding = kernel_size // 2

    x = F.conv3d(x, ky, padding=(0, padding, 0), groups=C)
    x = F.conv3d(x, kx, padding=(0, 0, padding), groups=C)

    # --- reshape back ---
    x = x.view(B, D, H_p, W_p, C, patch_size, patch_size)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)
    x = x.reshape(B, C, D, H, W)

    return x