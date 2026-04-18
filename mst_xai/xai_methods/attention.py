import torch
import torch.nn.functional as F
# import numpy as np

from mst_xai.xai_methods.base import BaseSaliencyMethod

# Import supported MST model architectures.
# This XAI method is designed to work with both CNN-based (ResNet)
# and Transformer-based (DINOv2) models that expose attention via
# get_attention_maps() and/or get_slice_attention(). or create attention_rollout().
# The imports define the intended scope of compatible models rather
# than being used explicitly in this file.
# from mst.models.resnet import ResNet, ResNetSliceTrans
# from mst.models.dino import DinoV2ClassifierSlice


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
        attention_method: "last_layer", "rollout", or "slice_weighted_rollout"

    Produces:
    - Spatial attention saliency aligned to input volume [D, H, W]
    - Or slice-level saliency [D]
    """

    def __init__(
        self,
        model,
        mode: str = "spatial",   # "spatial" or "slice"
        attention_method: str = "last_layer", 
        resize_to_input: bool = True,
    ):
        super().__init__(model)
        self.model.eval()  # Set model to evaluation mode

        assert mode in ["spatial", "slice"]
        assert attention_method in ["last_layer", "slice_weighted_rollout"]

        self.mode = mode
        self.resize_to_input = resize_to_input
        self.attention_method = attention_method

    # # --------------------------------------------------
    # # Detect number of extra tokens (CLS + register/storage)
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
        num_special = 4 # Storage tokens in DINOv3 (not include CLS token)
                
        # src_key_padding_mask = batch.get("src_key_padding_mask", None)
    
        # --------------------------------------------------
        # Forward pass (attention stored internally)
            # This forces the model to store attention matrices internally
            # Forward pass with attention recording enabled.
            # The model is expected to internally store attention matrices when
            # save_attn=True, which are later retrieved for explainability.

        # --------------------------------------------------
        _ = self.model( 
            source,
            # src_key_padding_mask=src_key_padding_mask,
            save_attn=True,
            # use_softmax=True,
        )
    
        # --------------------------------------------------
        # Retrieve attention from model
            # Retrieve attention from the model.
            # - attn_spatial: patch-level self-attention (e.g. ViT encoder)
            # - attn_slice: slice-level attention (e.g. slice fusion transformer)
            # At least one of these must be implemented by the model.

        # --------------------------------------------------
        attn_slice = self.model.get_slice_attention()       # [32,1,1]
        # attn_slice = attn_slice.squeeze(-1) # [32,1] - remove last dim

        #=---------------------------------------------
        # Attention Method Selection
        #---------------------------------------------

        if self.attention_method == "last_layer":
            #Final layer attention (default)

            attn_spatial = self.model.attention_maps[-1]   # [32, 12, 201, 201] [B*D, Heads, Tokens, Tokens] - take last layer attention
            attn_spatial = attn_spatial[:,:, 0, 1+num_special:] # CLS token attend to all tokens (remove extra tokens) # [B, num_heads, 1 , N-1-4]
            attn_spatial /= attn_spatial.sum(dim=-1, keepdim=True) # Normalize to make sum = 1 [B*D, Heads, N-1-4]
            cls_attn = attn_slice * attn_spatial # [B*D, Heads, N-1-4]
            cls_attn = cls_attn.mean(dim=1)  # Average over heads → [B*D, N-1-4] [32, 196]
            
        elif self.attention_method == "slice_weighted_rollout":
            # Slice-weighted attention rollout
            attn_maps = self.model.attention_maps  # list of [B*D, Heads, Tokens, Tokens]
            rollout = self._attention_rollout(attn_maps, num_special = num_special, discard_ratio=0.9)  # [B*D, N-1-4]
            slice_weights = attn_slice.squeeze(-1) # [B*D,1]
            cls_attn = slice_weights * rollout  # [32, 196]
        else:
            raise ValueError(f"Unknown attention method: {self.attention_method}")

    
        if cls_attn is None and attn_slice is None:
            raise RuntimeError("No attention maps found. Did you pass save_attn=True?")
    
        # --------------------------------------------------
        # Select mode
        # --------------------------------------------------
        if self.mode == "slice":
            # slice attention: [B, D] → [D]
            sal = attn_slice[0]

        else:

            # --------------------------------------------------
            # Spatial Mode
            # --------------------------------------------------
            B, C, D, H, W = source.shape

            patch_size = self.model.encoder.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]

            H_p = H // patch_size
            W_p = W // patch_size

            # cls_attn is [B*D, HW], reshape to [B, D, H_p, W_p]
            cls_attn = cls_attn.view(B, D, H_p, W_p)
            # Upsample each slice independently
            cls_attn = cls_attn.view(B * D, 1, H_p, W_p)

            sal = F.interpolate(
                cls_attn,
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
        sal = sal / (sal.max() - sal.min() + 1e-8)
    
        return sal.detach()


    # # --------------------------------------------------
    # # Visualization (same philosophy as GradCAM_MST)
    # # --------------------------------------------------
    # def visualize(
    #     self,
    #     image: torch.Tensor,
    #     saliency: torch.Tensor,
    #     alpha: float = 0.5,
    #     slice_idx: int | None = None,
    # ):
    #     """
    #     Args:
    #         image: [1, D, H, W]
    #         saliency:
    #             spatial → [D, H, W]
    #             slice   → [D]

    #     Returns:
    #         overlay (numpy)
    #     """

    #     img = image.squeeze().detach().cpu().numpy()
    #     sal = saliency.detach().cpu().numpy()

    #     if self.mode == "slice":
    #         if slice_idx is None:
    #             slice_idx = sal.argmax()
    #         return sal, slice_idx

    #     # spatial mode
    #     if slice_idx is None:
    #         slice_scores = sal.reshape(sal.shape[0], -1).sum(axis=1)
    #         slice_idx = slice_scores.argmax()

    #     img_slice = img[slice_idx]
    #     sal_slice = sal[slice_idx]

    #     sal_slice = (sal_slice - sal_slice.min()) / (sal_slice.max() + 1e-8)

    #     overlay = (1 - alpha) * img_slice + alpha * sal_slice
    #     overlay = np.clip(overlay, 0, 1)

    #     return overlay
    
    # def _attention_rollout(self, attn_maps):
    #     rollout = None
    #     for attn in attn_maps:
    #         # attn: [B, Heads, Tokens, Tokens]
    #         attn = attn.mean(dim=1)
    #         # Add residual connection
    #         I = torch.eye(attn.size(-1), device=attn.device)
    #         attn = attn + I
    #         # Normalize
    #         attn = attn / attn.sum(dim=-1, keepdim=True)
    #         rollout = attn if rollout is None else attn @ rollout
    #     return rollout[:, 0, 1:]  # CLS → patch attention # [B, HW]


    def _attention_rollout(self, attn_maps, num_special, discard_ratio):

        rollout = None

        for attn in attn_maps:
            # [B, Heads, N , N]
            attn = attn.max(dim=1).values  # Max over heads to get [B, N, N]
            # Discard lowes attention values (reduce noise)
            B, N, _ = attn.shape

            if discard_ratio > 0:
                k = int(N * discard_ratio)
                if k > 0:
                    # find indices of smallest values per row
                    vals, idx = torch.topk(attn, k=k, dim=-1, largest=False)
                    # build mask
                    mask = torch.zeros_like(attn, dtype=torch.bool)
                    mask.scatter_(-1, idx, True)
                    # protect CLS + special tokens (columns)
                    mask[:, :, :1 + num_special] = False
                    # apply mask
                    attn = attn.masked_fill(mask, 0)

            I = torch.eye(attn.size(-1), device=attn.device) # [N, N]
            
            I = I.unsqueeze(0) # [1, N, N]

            attn = attn + I # Add residual connection # [B, N, N]
            attn = attn / attn.sum(dim=-1, keepdim=True) # Normalize to make sum = 1 # [B, N, N]

            rollout = attn if rollout is None else torch.matmul(attn, rollout) # Recursive multiplication # [B, N, N] @ [B, N, N] → [B, N, N]
        # CLS → patches

        rollout = rollout[:, 0, 1+num_special:]   # [B, N]

        rollout = rollout / rollout.sum(dim=-1, keepdim=True) # [B, N] Normalize final rollout

        return rollout

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() - x.min() + 1e-8)
        return x