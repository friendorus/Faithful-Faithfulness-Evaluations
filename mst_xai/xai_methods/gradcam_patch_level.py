import torch
import torch.nn.functional as F
import numpy as np

from mst_xai.xai_methods.base import BaseSaliencyMethod


class GradCAM_MST(BaseSaliencyMethod):
    """
    Grad-CAM for Medical Slice Transformer (DinoV2ClassifierSlice)

    Produces:
    - Normalized saliency map aligned to input volume [D, H, W]
    - Optional visualization overlays
    """

    def __init__(self, model, target_layer="encoder"):
        super().__init__(model)
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    # --------------------------------------------------
    # Hooks
    # --------------------------------------------------
    def _register_hooks(self):
        """
        Hook the LAST DINO attention block (token-level)
        """
        self.gradients = None
        self.activations = None
    
        # Last transformer block
        block = self.model.encoder.blocks[-1]
    
        def forward_hook(module, input, output):
            # output: (B*D, N, C)
            self.activations = output
    
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: (B*D, N, C)
            self.gradients = grad_output[0]
    
        block.register_forward_hook(forward_hook)
        block.register_full_backward_hook(backward_hook)


    # --------------------------------------------------
    # Core Grad-CAM
    # --------------------------------------------------
    def generate(self, batch, target_class: int):
        self.model.zero_grad()
        source = batch["source"].to(self.model.device)
    
        B, C, D, H, W = source.shape
        patch_size = 14  # DINOv2
    
        logits = self.model(source, save_attn=False)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)
    
        # NOW shapes are correct
        acts = self.activations        # [(B*D), N+1, C]
        grads = self.gradients         # [(B*D), N+1, C]
    
        # Remove CLS token
        acts = acts[:, 1:, :]
        grads = grads[:, 1:, :]
    
        # Grad-CAM weights
        weights = grads.mean(dim=1)        # average over tokens → [(B*D), C]
        cam = torch.relu(
            (acts * weights.unsqueeze(1)).sum(dim=-1)
        )                                  # → [(B*D), N]

    
        # Patch grid
        h_p = H // patch_size
        w_p = W // patch_size
    
        cam = cam.view(B, D, h_p, w_p)
    
        # Upsample to voxel space
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(1)
    
        cam = cam.squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
    
        return cam



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
