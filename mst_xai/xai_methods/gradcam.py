import torch
import torch.nn.functional as F
# import numpy as np

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

from mst_xai.xai_methods.base import BaseSaliencyMethod

class GradCAM_MST(BaseSaliencyMethod):
    """
    Standard Grad-CAM for MST (ViT-based)

    Key points:
    - Computes gradients of target class score w.r.t. feature token activations of a specific layer.
    - Use gradients to weight the activations and produce a coarse saliency map.
    - Uses target layer at encoder.blocks[-1].norm1 (better than norm2 or layer.norm for ViT-based MST).
    - Returns volumetric saliency [D, H, W]
    """

    def __init__(
            self, 
            model,
            cam_method: str = "gradcam", #gradcam or hires_cam,
            relu: bool = True
            ):
        super().__init__(model)
        self.model.eval() # Set model to evaluation mode
        self.cam_method = cam_method
        self.relu = relu

        self.num_special = self._detect_num_special_tokens()

    def _detect_num_special_tokens(self):
        if hasattr(self, "num_special"):
            return self.num_special

        device = self.model.device
        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, 224, 224,
                device=device
            )
            out = self.model.encoder.forward_features(dummy)

        if "x_storage_tokens" in out: # DinoV3 (CLS + storage tokens
            self.num_special = out["x_storage_tokens"].shape[1]
        elif "x_norm_regtokens" in out: # DinoV2 (CLS + reg tokens)
            self.num_special = out["x_norm_regtokens"].shape[1]
        else:
            self.num_special = 0

        return self.num_special
    # --------------------------------------------------
    # Detect number of extra tokens (CLS + register/storage)
    # # --------------------------------------------------
    # def _get_num_extra_tokens(self, source):
    #     """
    #     This method detects how many extra tokens (beyond the CLS token) are present in the ViT encoder's output.        
    #     # Remove CLS + extra tokens (we only want patch tokens for spatial saliency)
    #     # ViT tokes = [CLS] + [extra tokens] + [patch tokens]
    #     # For DinoV2, there are possible to have only 1 CLS toke or extra register tokens,
    #     # For DinoV3, there are possible to have 1 CLS token + 4 storage tokens
    #     """
    #     if hasattr(self, "num_extra_tokens"):
    #         return self.num_extra_tokens

    #     # No gradient needed -> just inspect model structure to determine how many extra tokens there are (e.g., CLS + storage tokens)
    #     with torch.no_grad():
    #         # Take one slice for probing
    #         x_enc = source[:1]              # (1,1,D,H,W)
    #         x_enc = x_enc[:, :, 0]          # take one slice → (1,1,H,W)
    #         # Convert to 3-channel by repeating the single channel (ViT requires 3-channel input) → (1,3,H,W)
    #         x_enc = x_enc.repeat(1, 3, 1, 1)  # → (1,3,H,W)

    #         # Get token structure
    #         out = self.model.encoder.forward_features(x_enc)

    #     # Detect exttra tokens based on the output of the encoder's forward_features method.
    #     # If new models have different token structures, this logic may need to be updated.
    #     if "x_storage_tokens" in out:
    #         # DinoV3 (CLS + storage tokens)
    #         self.num_extra_tokens = out["x_storage_tokens"].shape[1] 
    #     elif "x_norm_regtokens" in out:
    #         # DinoV2 (CLS + normalized register tokens)
    #         self.num_extra_tokens = out["x_norm_regtokens"].shape[1]
    #     else:
    #         # Default to 0 if no extra tokens are detected (only CLS token)
    #         self.num_extra_tokens = 0

    #     return self.num_extra_tokens
    
    def _remove_extra_tokens(self, acts, grads):
        num_extra_tokens = self.num_special # For DINOv3, there are 4 storage tokens in addition to the CLS token. For DINOv2, there are normalized register tokens. Adjust as needed for different models.
        # Remove CLS and extra tokens from activations and gradients to focus on patch tokens
        acts = acts[:, 1 + num_extra_tokens:, :] # 1+num_extra_tokens to skip CLS and extra tokens
        grads = grads[:, 1 + num_extra_tokens:, :]
        return acts, grads

    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def _compute_cam(self, acts, grads):
        weights = grads.mean(dim=(0, 1)) # (C,) # Global average pooling of gradients across slices and tokens
        weights = weights.unsqueeze(0).unsqueeze(1)  # (B*D, 1, C) # Reshape for broadcasting to activations
        cam = (acts * weights).sum(dim=2)  # (B*D, N) # Linear combination of activations weighted by importance scores (gradients)
        if self.relu:
            cam = torch.relu(cam)
        return cam
    
    # --------------------------------------------------
    # HiResCAM
    # --------------------------------------------------    
    def _compute_hires_cam(self, acts, grads):
        # HiResCAM method: element-wise product of activations and gradients, then sum over channels
        cam = (acts * grads).sum(dim=2)  # (B*D, N) # Element-wise product followed by summation over channels
        if self.relu:
            cam = torch.relu(cam)
        return cam

    # -------------------------------------------------    

    def _generate_cam(self, source, target_class: int):
        # source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        ## We choose norm1 of the last block as the target layer for Grad-CAM, as it provides stronger gradients than norm2 or layer.norm in ViT-based MST.
        # target_layer = self.model.encoder.norm
        # target_layer = self.model.encoder.blocks[-1].norm2 
        self.target_layer = self.model.encoder.blocks[-1].norm1 # Already try norm2 and layer.norm but they provide zero grads.
        
        self.activations_and_grads = ActivationsAndGradients(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=None
            )

        B, C, D, H, W = source.shape
        patch_size = self.model.encoder.patch_embed.patch_size[0]
   
        source.requires_grad = True  
        logits = self.model(source)
        # logits = self.activations_and_grads(source)
        class_specific_logits = logits[:, target_class] 
    

        self.model.zero_grad()
        class_specific_logits.backward()  # Compute gradients

        acts = self.activations_and_grads.activations  # (B*D, N, C)
        grads = self.activations_and_grads.gradients   # (B*D, N, C)
        acts = acts[0]
        grads = grads[0]
        acts, grads = self._remove_extra_tokens(acts, grads) # Remove CLS and extra tokens

        if self.cam_method == "gradcam":
            cam = self._compute_cam(acts, grads)  # Compute CAM using manual method
        elif self.cam_method == "hires_cam":
            cam = self._compute_hires_cam(acts, grads)  # Compute HiResCAM using manual method
        else:
            raise ValueError(f"Unsupported method: {self.cam_method}")

        h_p = H // patch_size
        w_p = W // patch_size
        
        cam = cam.view(B, D, h_p, w_p)  # Reshape from (B*D, N) into volume (B, D, h_p, w_p)

        cam = F.interpolate(
            cam.unsqueeze(1),      # (B,1,D,h_p,w_p)
            size=(D, H, W),        # target size 
            mode="trilinear",
            align_corners=False
        ).squeeze(1)  # Upsample to voxel space

        cam = cam.squeeze(0)  
        cam = cam - cam.min()
        cam = cam / (cam.max() - cam.min() + 1e-8)  # Normalize

        self.activations_and_grads.release()  # Clean up hooks

        return cam.detach().cpu()
    
    
    # ----------------------------------
    # Main Entry
    # ----------------------------------
    def generate(self, batch, target_class: int):

        source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        return self._generate_cam(source, target_class)