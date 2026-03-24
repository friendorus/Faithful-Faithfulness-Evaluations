import torch
import torch.nn.functional as F
import numpy as np



from mst_xai.xai_methods.base import BaseSaliencyMethod


class GradCAM_MST(BaseSaliencyMethod):
    """
    Hierarchical Grad-CAM for Medical Slice Transformer

    Produces:
    - Slice-level importance
    - Spatial heatmap per slice
    - Combined final saliency aligned with final class decision
    """

    def __init__(self, model):
        super().__init__(model)

        # DINO (patch-level)
        self.patch_activations = None
        self.patch_gradients = None

        # Slice transformer (slice-level)
        self.slice_activations = None
        self.slice_gradients = None


        self._register_hooks()

    # --------------------------------------------------
    # Hooks
    # --------------------------------------------------
    def _register_hooks(self):

        # ---------- Patch-level hook ----------
        # patch_block = self.model.encoder.norm
        # patch_block = self.model.encoder.blocks[-1].norm2       
        patch_block = self.model.encoder.blocks[-1].norm1

        def forward_patch(module, input, output):
            self.patch_activations = output  # (B*D, N, C)

        def backward_patch(module, grad_input, grad_output):
            self.patch_gradients = grad_output[0]

        patch_block.register_forward_hook(forward_patch)
        patch_block.register_full_backward_hook(backward_patch)

        # ---------- Slice-level hook ----------
        # slice_block = self.model.slice_fusion.norm
        # slice_block = self.model.slice_fusion.layers[-1].norm2
        slice_block = self.model.slice_fusion.layers[-1].norm1

        def forward_slice(module, input, output):
            self.slice_activations = output  # (B, D, C)

        def backward_slice(module, grad_input, grad_output):
            self.slice_gradients = grad_output[0]

        slice_block.register_forward_hook(forward_slice)
        slice_block.register_full_backward_hook(backward_slice)


    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def generate(self, batch, target_class: int):
        #self.model

        self.model.zero_grad()
        source = batch["source"].to(self.model.device)

        B, C, D, H, W = source.shape
        #patch_size = 14 if using V2
        patch_size = self.model.encoder.patch_embed.patch_size[0]

        logits = self.model(source, save_attn=False)
        score = logits[:, target_class].sum()
        score.backward()


        # ===============================
        # PATCH-LEVEL CAM
        # ===============================

        acts = self.patch_activations      # (B*D, N, C) # Already test - not zero
        grads = self.patch_gradients


        # ----------------------------------
        # Remove CLS + extra tokens (robust)
        # ----------------------------------

        # detect extra tokens once
        if not hasattr(self, "num_extra_tokens"):
            with torch.no_grad():
                x_enc = source[:1]              # (1,1,D,H,W)
                x_enc = x_enc[:, :, 0]          # take one slice → (1,1,H,W)
                x_enc = x_enc.repeat(1, 3, 1, 1)  # → (1,3,H,W)

                out = self.model.encoder.forward_features(x_enc)

            if "x_storage_tokens" in out:
                self.num_extra_tokens = out["x_storage_tokens"].shape[1] #DinoV3 VitB having 5 extra tokens (CLS + 4 storage tokens)
            elif "x_norm_regtokens" in out:
                self.num_extra_tokens = out["x_norm_regtokens"].shape[1] #DinoV2 VitS having 5 extra tokens (CLS + 4 reg tokens) if it's use register
            else:
                self.num_extra_tokens = 0

        # remove CLS + extra tokens
        acts = acts[:, 1 + self.num_extra_tokens:, :]
        grads = grads[:, 1 + self.num_extra_tokens:, :]

        weights = grads.mean(dim=1)        # (B*D, C)

        cam_patch = (acts * weights.unsqueeze(1)).sum(dim=2)                              # (B*D, N_patches)

        h_p = H // patch_size
        w_p = W // patch_size

        cam_patch = cam_patch.view(B, D, h_p, w_p)


        # ===============================
        # SLICE-LEVEL CAM
        # ===============================

        # remove slice CLS token (there are 33 slices but only 32 have patch-level maps)
        slice_acts = self.slice_activations[:, 1:, :] # (B, 32, 284)
        slice_grads = self.slice_gradients[:, 1:, :]

        slice_weights = slice_grads.mean(dim=1)  # (B, C)

        cam_slice = (
            (slice_acts * slice_weights.unsqueeze(1)).sum(dim=-1)
        )                                        # (B, D)

        cam_slice = cam_slice / (cam_slice.max(dim=1, keepdim=True)[0] + 1e-8)
        assert cam_patch.shape[1] == cam_slice.shape[1]


        # ===============================
        # HIERARCHICAL COMBINATION
        # ===============================

        cam_slice = cam_slice.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)

        cam_combined = cam_patch * cam_slice               # weight spatial maps
        cam_combined = torch.relu(cam_combined)

        # ===============================
        # Upsample to voxel resolution
        # ===============================

        cam_combined = F.interpolate(
            cam_combined.unsqueeze(1),
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(1)

        cam_combined = cam_combined.squeeze(0)

        cam_combined = cam_combined - cam_combined.min()
        cam_combined = cam_combined / (cam_combined.max() + 1e-8)

        return cam_combined



    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def visualize(
        self,
        image: torch.Tensor,
        saliency: torch.Tensor,
        alpha: float = 0.5,
        slice_idx: int | None = None,
    ):
        """
        Create overlay visualization

        Args:
            image: [1, D, H, W]
            saliency: [D, H, W]
            slice_idx: optional slice index

        Returns:
            overlay: numpy image
        """
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

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x
