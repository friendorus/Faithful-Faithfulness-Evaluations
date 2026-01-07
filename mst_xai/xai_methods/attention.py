import torch
import torch.nn.functional as F
import numpy as np

from mst_xai.xai_methods.base import BaseSaliencyMethod
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoV2ClassifierSlice


class Attention_MST(BaseSaliencyMethod):
    """
    Attention-based saliency for MST-style models.

    Produces:
    - Spatial attention saliency aligned to input volume [D, H, W]
    - Or slice-level saliency [D]
    """

    def __init__(
        self,
        model,
        mode: str = "spatial",   # "spatial" or "slice"
        resize_to_input: bool = True,
    ):
        super().__init__(model)
        assert mode in ["spatial", "slice"]
        self.mode = mode
        self.resize_to_input = resize_to_input

    # --------------------------------------------------
    # Core Attention Saliency
    # --------------------------------------------------
    def generate(self, batch, target_class=None):
        """
        Returns:
            spatial mode -> [D, H, W]
            slice mode   -> [D]
        """
    
        source = batch["source"].to(self.model.device)
        src_key_padding_mask = batch.get("src_key_padding_mask", None)
    
        # --------------------------------------------------
        # Forward pass (attention stored internally)
        # --------------------------------------------------
        _ = self.model(
            source,
            src_key_padding_mask=src_key_padding_mask,
            save_attn=True,
            use_softmax=True,
        )
    
        # --------------------------------------------------
        # Retrieve attention from model
        # --------------------------------------------------
        attn_spatial = self.model.get_attention_maps()      # [B, heads, tokens, tokens]
        attn_slice = self.model.get_slice_attention()       # [B, D]
    
        if attn_spatial is None and attn_slice is None:
            raise RuntimeError("No attention maps found. Did you pass save_attn=True?")
    
        # --------------------------------------------------
        # Select mode
        # --------------------------------------------------
        if self.mode == "slice":
            # slice attention: [B, D] → [D]
            sal = attn_slice[0]
    
        else:
            # spatial attention
            # expected shape: [B, heads, tokens, tokens]
            # use CLS → patch attention
            # attn_spatial: [B, heads, H_p * W_p]
            cls_attn = attn_spatial.mean(dim=1)  # [B, N]
            
            B, C, D, H, W = source.shape
            patch_size = self.model.encoder.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            
            H_p = H // patch_size
            W_p = W // patch_size
            N_expected = H_p * W_p
            
            if cls_attn.shape[1] != N_expected:
                raise RuntimeError(
                    f"Patch count mismatch: expected {N_expected}, got {cls_attn.shape[1]}"
                )
            
            # reshape 2D patch grid
            sal_2d = cls_attn[0].reshape(H_p, W_p)
            
            # upsample to full resolution
            sal_2d = F.interpolate(
                sal_2d.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )[0, 0]
            
            # broadcast across slices
            sal = sal_2d.unsqueeze(0).repeat(D, 1, 1)  # [D, H, W]

    
        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        sal = sal - sal.min()
        sal = sal / (sal.max() + 1e-8)
    
        return sal.detach()


    # --------------------------------------------------
    # Visualization (same philosophy as GradCAM_MST)
    # --------------------------------------------------
    def visualize(
        self,
        image: torch.Tensor,
        saliency: torch.Tensor,
        alpha: float = 0.5,
        slice_idx: int | None = None,
    ):
        """
        Args:
            image: [1, D, H, W]
            saliency:
                spatial → [D, H, W]
                slice   → [D]

        Returns:
            overlay (numpy)
        """

        img = image.squeeze().detach().cpu().numpy()
        sal = saliency.detach().cpu().numpy()

        if self.mode == "slice":
            if slice_idx is None:
                slice_idx = sal.argmax()
            return sal, slice_idx

        # spatial mode
        if slice_idx is None:
            slice_scores = sal.reshape(sal.shape[0], -1).sum(axis=1)
            slice_idx = slice_scores.argmax()

        img_slice = img[slice_idx]
        sal_slice = sal[slice_idx]

        sal_slice = (sal_slice - sal_slice.min()) / (sal_slice.max() + 1e-8)

        overlay = (1 - alpha) * img_slice + alpha * sal_slice
        overlay = np.clip(overlay, 0, 1)

        return overlay


    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x
