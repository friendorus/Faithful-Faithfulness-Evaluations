import torch
import torch.nn.functional as F
import numpy as np

from mst_xai.xai_methods.base import BaseSaliencyMethod


class GradCAM_MST(BaseSaliencyMethod):
    """
    Standard Grad-CAM for MST (ViT-based)

    - Uses target layer at encoder.blocks[-1].norm1
    - Returns volumetric saliency [D, H, W]
    """

    def __init__(self, model):
        super().__init__(model)

        self.activations = None
        self.gradients = None

        self._register_hooks()

    # --------------------------------------------------
    # Hook: Last ViT block (norm1)
    # --------------------------------------------------
    def _register_hooks(self):
        # target_layer = self.model.encoder.norm
        # target_layer = self.model.encoder.blocks[-1].norm2 
        target_layer = self.model.encoder.blocks[-1].norm1 # Already try norm2 and layer.norm but they provide zero grads.

        def forward_hook(module, input, output):
            self.activations = output  # (B*D, N, C)

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # (B*D, N, C)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def generate(self, batch, target_class: int):

        self.model.zero_grad()

        source = batch["source"].to(self.model.device)  # (B, C, D, H, W)
        
        B, C, D, H, W = source.shape

        patch_size = self.model.encoder.patch_embed.patch_size[0] # 14 for V2, 16 for V3

        # Forward
        logits = self.model(source)
        score = logits[:, target_class].sum()

        # Backward
        score.backward()

        acts = self.activations      # (B*D, N, C)
        grads = self.gradients       # (B*D, N, C)

        # --------------------------------------------------
        # Remove CLS + extra tokens
        # For DinoV2, there are possible to have only 1 CLS toke or extra register tokens,
        # For DinoV3, there are possible to have 1 CLS token + 4 storage tokens
        # --------------------------------------------------
        if not hasattr(self, "num_extra_tokens"):
            with torch.no_grad():
                x_enc = source[:1]              # (1,1,D,H,W)
                x_enc = x_enc[:, :, 0]          # take one slice → (1,1,H,W)
                x_enc = x_enc.repeat(1, 3, 1, 1)  # → (1,3,H,W)

                out = self.model.encoder.forward_features(x_enc)

            if "x_storage_tokens" in out:
                self.num_extra_tokens = out["x_storage_tokens"].shape[1]
            elif "x_norm_regtokens" in out:
                self.num_extra_tokens = out["x_norm_regtokens"].shape[1]
            else:
                self.num_extra_tokens = 0

        acts = acts[:, 1 + self.num_extra_tokens:, :]
        grads = grads[:, 1 + self.num_extra_tokens:, :]

        # --------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------
        weights = grads.mean(dim=1)  # (B*D, C)

        cam = (acts * weights.unsqueeze(1)).sum(dim=2)  # (B*D, N)
        cam = torch.relu(cam)

        # --------------------------------------------------
        # Reshape to spatial grid
        # --------------------------------------------------
        h_p = H // patch_size
        w_p = W // patch_size

        cam = cam.view(B, D, h_p, w_p)

        # --------------------------------------------------
        # Upsample to voxel space
        # --------------------------------------------------
        cam = F.interpolate(
            cam.unsqueeze(1),                 # (B,1,D,h_p,w_p)
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(1)

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze(0)  # (D, H, W)

    # --------------------------------------------------
    # Visualization (optional helper)
    # --------------------------------------------------
    def visualize(self, image, saliency, alpha=0.5, slice_idx=None):
        img = image.squeeze().detach().cpu().numpy()
        sal = saliency.detach().cpu().numpy()

        if slice_idx is None:
            slice_scores = sal.reshape(sal.shape[0], -1).sum(axis=1)
            slice_idx = slice_scores.argmax()

        img_slice = img[slice_idx]
        sal_slice = sal[slice_idx]

        sal_slice = (sal_slice - sal_slice.min()) / (sal_slice.max() + 1e-8)

        overlay = (1 - alpha) * img_slice + alpha * sal_slice
        overlay = np.clip(overlay, 0, 1)

        return overlay