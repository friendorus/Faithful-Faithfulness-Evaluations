import torch
import torch.nn.functional as F
import numpy as np

from mst_xai.xai_methods.base import BaseSaliencyMethod

# Import supported MST model architectures.
# This XAI method is designed to work with both CNN-based (ResNet)
# and Transformer-based (DINOv2) models that expose attention via
# get_attention_maps() and/or get_slice_attention(). or create attention_rollout().
# The imports define the intended scope of compatible models rather
# than being used explicitly in this file.
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoV2ClassifierSlice


class Attention_MST(BaseSaliencyMethod): #Attention-based saliency (CLS-to-patch) or Attention Rollout_MST
    """
    Attention-based saliency for MST-style models.

    This method extracts attention weights stored during the forward pass
    and converts them into saliency maps. Depending on the mode, it produces:
    - spatial saliency maps based on CLS-to-patch attention (ViT-style models)
        # Select between last-layer attention and attention rollout.
        # Attention rollout propagates attention across all Transformer encoder layers
        # while the default uses only the final layer.
    - slice-level importance scores based on slice-attention mechanisms

    Args:
        model: Transformer-based MST model with attention layers
        mode: "spatial" produces [D, H, W], "slice" produces [D]
        resize_to_input: If True, upsamples spatial saliency to input resolution
        use_rollout: If True, uses attention rollout for spatial attention (default: False)

    Produces:
    - Spatial attention saliency aligned to input volume [D, H, W]
    - Or slice-level saliency [D]
    """

    def __init__(
        self,
        model,
        mode: str = "spatial",   # "spatial" or "slice"
        resize_to_input: bool = True,
        use_rollout: bool = False,  # NEW
    ):
        super().__init__(model)
        assert mode in ["spatial", "slice"]
        self.mode = mode
        self.resize_to_input = resize_to_input
        self.use_rollout = use_rollout  # NEW

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
            # This forces the model to store attention matrices internally
            # Forward pass with attention recording enabled.
            # The model is expected to internally store attention matrices when
            # save_attn=True, which are later retrieved for explainability.

        # --------------------------------------------------
        _ = self.model( 
            source,
            src_key_padding_mask=src_key_padding_mask,
            save_attn=True,
            use_softmax=True,
        )
    
        # --------------------------------------------------
        # Retrieve attention from model
            # Retrieve attention from the model.
            # - attn_spatial: patch-level self-attention (e.g. ViT encoder)
            # - attn_slice: slice-level attention (e.g. slice fusion transformer)
            # At least one of these must be implemented by the model.

        # --------------------------------------------------
        attn_slice = self.model.get_slice_attention()       # [B, D] # slice attention # From dino.py
        
        if self.use_rollout:
            attn_maps = self.model.attention_maps  # list of [B*D, Heads, T, T]
            attn_spatial = self._attention_rollout(attn_maps)  # [B*D, HW]
            print(f"Attention rollout applied. Spatial attention shape: {attn_spatial.shape}")

            # ---- Get slice weights ----
            slice_weights = self.model.get_slice_attention()  # [B*D, 1, 1]
            print(f"Slice weights shape (raw): {slice_weights.shape}")

            slice_weights = slice_weights.view(-1, 1)  # [B*D, 1]

            # ---- Apply slice weighting ----
            attn_spatial = attn_spatial * slice_weights
        else:
            attn_spatial = self.model.get_attention_maps()  # [B, heads, tokens, tokens] - only last layer (already combined with slice attention in dino.py)
    
        if attn_spatial is None and attn_slice is None:
            raise RuntimeError("No attention maps found. Did you pass save_attn=True?")
    
        # --------------------------------------------------
        # Select mode
        # --------------------------------------------------
        if self.mode == "slice":
            # slice attention: [B, D] → [D]
            sal = attn_slice[0]
            print(f"Slice attention saliency shape: {sal.shape}")
    
        else:
            # attn_spatial is [B*D, HW] (rollout case)
            attn = attn_spatial  # already [32, 256]

            B, C, D, H, W = source.shape

            patch_size = self.model.encoder.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]

            H_p = H // patch_size
            W_p = W // patch_size

            # Restore slice dimension
            # [B*D, HW] -> [B, D, H_p, W_p]
            attn = attn.view(B, D, H_p, W_p)

            # Upsample each slice independently
            attn = attn.view(B * D, 1, H_p, W_p)

            sal = F.interpolate(
                attn,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

            sal = sal.view(B, D, H, W)[0]
    
        # --------------------------------------------------
        # Normalize
            # Min–max normalization to [0, 1] for numerical stability,
            # visualization, and compatibility with insertion/deletion metrics.
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
    
    def _attention_rollout(self, attn_maps):
        rollout = None
        for attn in attn_maps:
            # attn: [B, Heads, Tokens, Tokens]
            attn = attn.mean(dim=1)
            # Add residual connection
            I = torch.eye(attn.size(-1), device=attn.device)
            attn = attn + I
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn if rollout is None else attn @ rollout
        return rollout[:, 0, 1:]  # CLS → patch attention # [B, HW]


    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x