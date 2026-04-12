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

    def __init__(self, model, mode="manual"):
        super().__init__(model)
        self.model.eval() # Set model to evaluation mode
        
        assert mode in ["manual", "library_encoder", "hybrid"], "Invalid mode. Choose from 'manual', 'library_encoder', or 'hybrid'."
        self.mode = mode

        ## We choose norm1 of the last block as the target layer for Grad-CAM, as it provides stronger gradients than norm2 or layer.norm in ViT-based MST.
        # target_layer = self.model.encoder.norm
        # target_layer = self.model.encoder.blocks[-1].norm2 
        self.target_layer = self.model.encoder.blocks[-1].norm1 # Already try norm2 and layer.norm but they provide zero grads.
        
        ## Manual and Hybrid modes Storage
        # Will store forward activations (features) and backward gradients (importance scores)
        self.activations = None
        self.gradients = None
        
        if mode == "manual":
            # Register hooks to capture activations and gradients from the target layer when class is created
            self._register_hooks()

        if mode == "hybrid":
            self.activations_and_grads = ActivationsAndGradients(
                model=self.model,
                target_layers=[self.target_layer],
                reshape_transform=None
                )

    # --------------------------------------------------
    # Hook: Last ViT block (norm1) for manual mode
    # --------------------------------------------------
    def _register_hooks(self):
  
        # Forward hook to capture activations (features) during forward pass
        def forward_hook(module, input, output):
            self.activations = output  # (B*D, N, C)

        # Backward hook to capture gradients during backward pass
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] = d(score)/d(output_of_target_layer)
            self.gradients = grad_output[0]  # (B*D, N, C)

        #register hooks into the model
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

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
    
    def _remove_extra_tokens(self, acts, grads, source):
        num_extra_tokens = self._get_num_extra_tokens(source)
        # Remove CLS and extra tokens from activations and gradients to focus on patch tokens
        acts = acts[:, 1 + num_extra_tokens:, :] # 1+num_extra_tokens to skip CLS and extra tokens
        grads = grads[:, 1 + num_extra_tokens:, :]
        return acts, grads

    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def _compute_cam(self, acts, grads):
        weights = grads.mean(dim=1)  # (B*D, C) #(32, 768) # Global average pooling of gradients across tokens
        weights = weights.unsqueeze(1)  # (B*D, 1, C) # Reshape for broadcasting to activations 
        cam = (acts * weights).sum(dim=2)  # (B*D, N) # Linear combination of activations weighted by importance scores (gradients)
        cam = torch.relu(cam)  
        return cam

    # --------------------------------------------------
    # Mode 1: Manual Grad-CAM for MST
    # --------------------------------------------------
    def generate(self, source, target_class: int):
        self.model.zero_grad() # Reset stored activations and gradients in model
        
        # source = batch["source"].to(self.model.device)  # (B, C, D, H, W)
        
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

        acts, grads = self._remove_extra_tokens(acts, grads, source)

        # --------------------------------------------------
        # Grad-CAM computation
        # --------------------------------------------------
        cam = self._compute_cam(acts, grads)  # (B*D, N)

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
    # Mode 2: Library (Encoder only) not whole MST model
    # --------------------------------------------------
    def generate_library_encoder(self, source, target_class: int):
        # source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        B, C, D, H, W = source.shape
        patch_size = self.model.encoder.patch_embed.patch_size[0]

        input_2d = source.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # (B*D, C, H, W) # Flatten 3D volume into 2D slices for Grad-CAM library
        # Repeat channel to match 3-channel input expected by ViT
        if input_2d.shape[1] == 1:
            input_2d = input_2d.repeat(1, 3, 1, 1)

        self.num_extra_tokens = self._get_num_extra_tokens(source) # Detect number of extra tokens for reshape transform
        h_p = H // patch_size
        w_p = W // patch_size

        def reshape_transform(tensor):
            tensor = tensor[:, 1+ self.num_extra_tokens:, :]  # Remove CLS and extra tokens
            tensor = tensor.reshape(B * D, h_p, w_p, -1)  # Reshape to spatial grid # -1 infers the channel dimension
            tensor = tensor.permute(0, 3, 1, 2) # Convert to (B*D, C, H, W) for Grad-CAM
            return tensor
        
        cam = GradCAM(
            model=self.model.encoder, # Only pass the encoder to the Grad-CAM library since we're targeting a layer within the encoder
            target_layers=[self.target_layer],
            reshape_transform=reshape_transform
        )

        targets = [ClassifierOutputTarget(target_class)] * (B * D) # Define targets for each slice
        cam = cam(input_tensor=input_2d, targets=targets)  # (B*D, H, W)
        cam = cam.reshape(B, D, H, W)  # Reshape back to volume
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return torch.tensor(cam.squeeze(0))  # (D, H, W)
    
    # --------------------------------------------------
    # Mode 3: Hybrid - Use library for activations and gradients, but manual computation for CAM
    # --------------------------------------------------
    def generate_hybrid(self, source, target_class: int):
        # source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        B, C, D, H, W = source.shape
        patch_size = self.model.encoder.patch_embed.patch_size[0]
   

        logits = self.model(source)  
        class_specific_logits = logits[:, target_class] 

        self.model.zero_grad()
        class_specific_logits.backward()  # Compute gradients

        acts = self.activations_and_grads.activations  # (B*D, N, C)
        grads = self.activations_and_grads.gradients   # (B*D, N, C)
        acts, grads = self._remove_extra_tokens(acts, grads, source) # Remove CLS and extra tokens

        cam = self._compute_cam(acts, grads)  # Compute CAM using manual method

        h_p = H // patch_size
        w_p = W // patch_size
        
        cam = cam.view(B, D, h_p, w_p)  # Reshape from (B*D, N) into volume (B, D, h_p, w_p)

        cam = F.interpolate(
            cam.unsqueeze(1),      # (B,1,D,h_p,w_p)
            size=(D, H, W),        # target size 
            mode="trilinear",
            align_corners=False
        ).squeeze(1)  # Upsample to voxel space

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # Normalize

        return cam.squeeze(0)  # (D, H, W)
    
    
    # ----------------------------------
    # Main Entry
    # ----------------------------------
    def generate(self, batch, target_class: int):

        source = batch["source"].to(self.model.device)  # (B, C, D, H, W)

        if self.mode == "manual":
            return self.generate_manual(source, target_class)
        elif self.mode == "library_encoder":
            return self.generate_library_encoder(source, target_class)
        elif self.mode == "hybrid":
            return self.generate_hybrid(source, target_class)
