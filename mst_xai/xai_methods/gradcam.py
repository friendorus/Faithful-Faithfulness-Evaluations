import torch
import torch.nn.functional as F
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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

    def __init__(self, model):
        super().__init__(model)
        # Will store forward activations (features) and backward gradients (importance scores)
        self.activations = None
        self.gradients = None
        # Register hooks to capture activations and gradients from the target layer when class is created
        self._register_hooks()

    # --------------------------------------------------
    # Hook: Last ViT block (norm1)
    # --------------------------------------------------
    def _register_hooks(self):
        # We choose norm1 of the last block as the target layer for Grad-CAM, as it provides stronger gradients than norm2 or layer.norm in ViT-based MST.
        # target_layer = self.model.encoder.norm
        # target_layer = self.model.encoder.blocks[-1].norm2 
        target_layer = self.model.encoder.blocks[-1].norm1 # Already try norm2 and layer.norm but they provide zero grads.
        
        # Forward hook to capture activations (features) during forward pass
        def forward_hook(module, input, output):
            self.activations = output  # (B*D, N, C)

        # Backward hook to capture gradients during backward pass
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] = d(score)/d(output_of_target_layer)
            self.gradients = grad_output[0]  # (B*D, N, C)

        #register hooks into the model
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    # --------------------------------------------------
    # Detect number of extra tokens (CLS + register/storage)
    # --------------------------------------------------
    def _get_num_extra_tokens(self, source):
        """
        This method detects how many extra tokens (beyond the CLS token) are present in the ViT encoder's output.        
        # Remove CLS + extra tokens (we only want patch tokens for spatial saliency)
        # ViT tokes = [CLS] + [extra tokens] + [patch tokens]
        # For DinoV2, there are possible to have only 1 CLS toke or extra register tokens,
        # For DinoV3, there are possible to have 1 CLS token + 4 storage tokens
        """
        if hasattr(self, "num_extra_tokens"):
            return self.num_extra_tokens

        # No gradient needed -> just inspect model structure to determine how many extra tokens there are (e.g., CLS + storage tokens)
        with torch.no_grad():
            # Take one slice for probing
            x_enc = source[:1]              # (1,1,D,H,W)
            x_enc = x_enc[:, :, 0]          # take one slice → (1,1,H,W)
            # Convert to 3-channel by repeating the single channel (ViT requires 3-channel input) → (1,3,H,W)
            x_enc = x_enc.repeat(1, 3, 1, 1)  # → (1,3,H,W)

            # Get token structure
            out = self.model.encoder.forward_features(x_enc)

        # Detect exttra tokens based on the output of the encoder's forward_features method.
        # If new models have different token structures, this logic may need to be updated.
        if "x_storage_tokens" in out:
            # DinoV3 (CLS + storage tokens)
            self.num_extra_tokens = out["x_storage_tokens"].shape[1] 
        elif "x_norm_regtokens" in out:
            # DinoV2 (CLS + normalized register tokens)
            self.num_extra_tokens = out["x_norm_regtokens"].shape[1]
        else:
            # Default to 0 if no extra tokens are detected (only CLS token)
            self.num_extra_tokens = 0

        return self.num_extra_tokens

    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def generate(self, batch, target_class: int):
        # Reset stored activations and gradients in model
        self.model.zero_grad()
        
        source = batch["source"].to(self.model.device)  # (B, C, D, H, W)
        
        B, C, D, H, W = source.shape

        # Patch size of ViT (e.g., 14 for DinoV2, 16 for DinoV3) by getting it from the model's encoder configuration.
        # This is needed to reshape the token activations back to spatial dimensions.
        patch_size = self.model.encoder.patch_embed.patch_size[0] # 14 for V2, 16 for V3

        # -------------------------
        # Forward pass
        # -------------------------
        logits = self.model(source)
        class_specific_logits = logits[:, target_class]


        # -------------------------
        # Backward pass
        # -------------------------
        # This computes gradients for ALL layers
        # Capture gradients at target layer via backward hook.
        class_specific_logits.backward()


        # Retrieve the stored activations and gradients from the hooks
        acts = self.activations      # (B*D, N, C)
        grads = self.gradients       # (B*D, N, C)

        num_extra_tokens = self._get_num_extra_tokens(source)  # Detect number of extra tokens (CLS + register/storage)

        # Remove CLS and extra tokens from activations and gradients to focus on patch tokens
        acts = acts[:, 1 + num_extra_tokens:, :]
        grads = grads[:, 1 + num_extra_tokens:, :]

        # --------------------------------------------------
        # Grad-CAM computation
        # --------------------------------------------------

        # Step 1: Global average pooling of gradients across the token dimension to get importance weights for each channel.
        # Equivalent to global average pooling in the original Grad-CAM paper, but applied to the token dimension instead of spatial dimensions.
        weights = grads.mean(dim=1)  # (B*D, C)

        # Step 2: Weight combination of activations with the importance weights to get the coarse saliency map.
        cam = (acts * weights.unsqueeze(1)).sum(dim=2)  # (B*D, N)

        # Step 3: Apply ReLU to focus on positive contributions (as per Grad-CAM paper).
        cam = torch.relu(cam)

        # --------------------------------------------------
        # Reshape tokens to spatial grid
        # --------------------------------------------------
        # Number of patches along height and width dimensions after removing CLS and extra tokens.
        h_p = H // patch_size
        w_p = W // patch_size

        # Reshape cam from (B*D, N) into volume (B, D, h_p, w_p)
        cam = cam.view(B, D, h_p, w_p)

        # --------------------------------------------------
        # Upsample to voxel space
        # --------------------------------------------------
        cam = F.interpolate(
            cam.unsqueeze(1),      # (B,1,D,h_p,w_p)
            size=(D, H, W),        # target size 
            mode="trilinear",
            align_corners=False
        ).squeeze(1)

        # --------------------------------------------------
        # Normalize to [0,1] 
        # --------------------------------------------------
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Return the saliency map for the batch (B, D, H, W). The caller can then select specific slices or aggregate as needed.
        return cam.squeeze(0)  # (D, H, W)

    # --------------------------------------------------
    # Visualization (optional helper)
    # --------------------------------------------------
    def visualize(self, image, saliency, alpha=0.5, slice_idx=None):
        # Convert tensors to numpy for visualization
        img = image.squeeze().detach().cpu().numpy()
        sal = saliency.detach().cpu().numpy()
        # Automatically pick the slice with the highest saliency if slice_idx is not provided
        if slice_idx is None:
            slice_scores = sal.reshape(sal.shape[0], -1).sum(axis=1)
            slice_idx = slice_scores.argmax()

        img_slice = img[slice_idx]
        sal_slice = sal[slice_idx]

        # Normalize the saliency slice to [0,1] for better visualization
        sal_slice = (sal_slice - sal_slice.min()) / (sal_slice.max() + 1e-8)

        # Overlay the saliency map on the original image slice using a simple alpha blending
        overlay = (1 - alpha) * img_slice + alpha * sal_slice
        overlay = np.clip(overlay, 0, 1)

        return overlay



class GradCAM_Library_MST(BaseSaliencyMethod):
    """
    Grad-CAM using pytorch-grad-cam library for MST (ViT-based)

    Key adaptations:
    - Flatten 3D volume into 2D slices
    - Reshape ViT tokens → spatial grid
    - Use encoder.blocks[-1].norm1 as target layer
    """

    def __init__(self, model):
        super().__init__(model)
        self.model.eval()

        # Target layer for Grad-CAM (ViT best practice)
        self.target_layers = [self.model.encoder.blocks[-1].norm1]

        # Cache for token handling
        self.num_extra_tokens = None

    # --------------------------------------------------
    # Detect number of extra tokens (CLS + register/storage)
    # --------------------------------------------------
    def _get_num_extra_tokens(self, source):
        if self.num_extra_tokens is not None:
            return self.num_extra_tokens

        with torch.no_grad():
            x = source[:1, :, 0]         # (1,1,H,W)
            x = x.repeat(1, 3, 1, 1)     # (1,3,H,W)

            out = self.model.encoder.forward_features(x)

        if "x_storage_tokens" in out:
            self.num_extra_tokens = out["x_storage_tokens"].shape[1]
        elif "x_norm_regtokens" in out:
            self.num_extra_tokens = out["x_norm_regtokens"].shape[1]
        else:
            self.num_extra_tokens = 0

        return self.num_extra_tokens

    # --------------------------------------------------
    # ViT reshape: tokens → feature map
    # --------------------------------------------------
    def _reshape_transform(self, tensor, height, width, num_extra_tokens):
        # tensor: (B*D, N, C)

        # Remove CLS + extra tokens
        tensor = tensor[:, 1 + num_extra_tokens:, :]

        B_D, N, C = tensor.shape

        # Convert to spatial grid
        tensor = tensor.reshape(B_D, height, width, C)

        # Convert to (B*D, C, H, W) for Grad-CAM
        tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def generate(self, batch, target_class: int):

        source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        B, C, D, H, W = source.shape
        patch_size = self.model.encoder.patch_embed.patch_size[0]

        # --------------------------------------------------
        input_tensor = source  # (B, C, D, H, W)

        # --------------------------------------------------
        # Token handling
        # --------------------------------------------------
        num_extra_tokens = self._get_num_extra_tokens(source)

        h_p = H // patch_size
        w_p = W // patch_size

        # --------------------------------------------------
        # Create Grad-CAM object
        # --------------------------------------------------
        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            reshape_transform=lambda x: self._reshape_transform(
                x, h_p, w_p, num_extra_tokens
            )
        )

        # --------------------------------------------------
        # Define targets (same class for all slices)
        # --------------------------------------------------
        targets = [ClassifierOutputTarget(target_class)] * (B * D)

        # --------------------------------------------------
        # Run Grad-CAM
        # --------------------------------------------------
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )  # (B*D, H, W)

        # --------------------------------------------------
        # Reshape back to volume
        # --------------------------------------------------
        cam_volume = grayscale_cam.reshape(B, D, H, W)

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        cam_volume = cam_volume - cam_volume.min()
        cam_volume = cam_volume / (cam_volume.max() + 1e-8)

        return torch.tensor(cam_volume.squeeze(0))

    # --------------------------------------------------
    # Visualization helper (same as before)
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