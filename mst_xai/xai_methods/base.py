# mst_xai/xai_methods/base.py
from abc import ABC, abstractmethod

class BaseSaliencyMethod(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, batch, target_class):
        """
        Returns:
            saliency: torch.Tensor [D, H, W] or [num_patches]
        """
        pass