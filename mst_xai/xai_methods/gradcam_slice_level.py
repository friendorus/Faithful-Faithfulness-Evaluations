import torch

class GradCAM_Slice:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.slice_fusion

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out: [B, N, C]
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0]: [B, N, C]
        self.gradients = grad_out[0]

    def generate(self, batch, target_class):
        self.model.zero_grad()

        logits = self.model(batch["source"])
        score = logits[:, target_class].sum()
        score.backward()

        acts = self.activations[:, 1:, :]   # remove CLS → [B, D, C]
        grads = self.gradients[:, 1:, :]    # [B, D, C]

        # slice-level Grad-CAM
        weights = grads.mean(dim=-1)        # [B, D]
        cam = torch.relu(weights * acts.mean(dim=-1))  # [B, D]

        # normalize per volume
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)

        return cam.squeeze(0)  # [D]
