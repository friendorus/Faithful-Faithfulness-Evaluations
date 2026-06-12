import torch
import torch.nn.functional as F

from mst_xai.xai_methods.base import BaseSaliencyMethod

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
        assert attention_method in ["last_layer", 
                                    "slice_weighted_rollout", "grad_sam", "grad_rollout",
                                    "nonclass_grad_sam", "nonclass_grad_rollout"]

        self.mode = mode
        self.resize_to_input = resize_to_input
        self.attention_method = attention_method
        self.num_special = self._detect_num_special_tokens()

    def _detect_num_special_tokens(self):
        if hasattr(self, "num_special"):
            return self.num_special

        device = self.model.device
        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, 224, 224,
                device=device
            )
            out = self.model.encoder.forward_features(dummy)

        if "x_storage_tokens" in out: # DinoV3 (CLS + storage tokens
            self.num_special = out["x_storage_tokens"].shape[1]
        elif "x_norm_regtokens" in out: # DinoV2 (CLS + reg tokens)
            self.num_special = out["x_norm_regtokens"].shape[1]
        else:
            self.num_special = 0

        return self.num_special
    # --------------------------------------------------
    # Core Attention Saliency
    # --------------------------------------------------
    def generate(self, batch, logits =None, target_class=None): 
        """
        Returns:
            spatial mode -> [D, H, W]
            slice mode   -> [D]
        """
    
        source = batch["source"].to(self.model.device)

        #=---------------------------------------------
        # Attention Method Selection
        #---------------------------------------------

        if self.attention_method == "last_layer":
            attn_spatial = self.model.attention_maps[-1]   # [32, 12, 201, 201] [B*D, Heads, Tokens, Tokens] - take last layer attention
            attn_spatial = attn_spatial[:,:, 0, 1+self.num_special:] # CLS token attend to all tokens (remove extra tokens) # [B, num_heads, 1 , N-1-4]
            attn_spatial /= attn_spatial.sum(dim=-1, keepdim=True) # Normalize to make sum = 1 [B*D, Heads, N-1-4]
            
            attn_slice = self.model.get_slice_attention()       # [32,1,1]
            cls_attn = attn_slice * attn_spatial # [B*D, Heads, N-1-4]
            cls_attn = cls_attn.mean(dim=1)  # Average over heads → [B*D, N-1-4] [32, 196]
            
        elif self.attention_method == "slice_weighted_rollout":
            attn_maps = self.model.attention_maps  # list of [B*D, Heads, Tokens, Tokens]
            rollout = self._attention_rollout(attn_maps, num_special = self.num_special)  # [B*D, N-1-4]
            
            attn_slice = self.model.get_slice_attention()       # [32,1,1]
            slice_weights = attn_slice.squeeze(-1) # [B*D,1]
            cls_attn = slice_weights * rollout  # [32, 196]

        elif self.attention_method in ["grad_sam","nonclass_grad_sam"]:
            if self.attention_method == "grad_sam":
                if target_class is None:
                    target_class = logits.argmax(dim=-1)
                score = logits[:, target_class]
                for attn in self.model.attention_maps:
                    attn.retain_grad() # command to retain gradients for every transformer layer
                self.model.zero_grad() # clear all gradient before backward pass
                score.backward()

                patch_attn_maps = self.model.attention_maps # list of [B*D, Heads, Tokens, Tokens]
                patch_cam = self._grad_sam(patch_attn_maps, num_special=self.num_special)  # [B*D, N-1-4]
                            
                # attn_slice = self.model.get_slice_attention()       # [32,1,1]
                # slice_weights = attn_slice.squeeze(-1) # [B*D,1]
                # cls_attn = slice_weights * patch_cam  # [32, 196]

                cls_attn = patch_cam

            elif self.attention_method == "nonclass_grad_sam":
                num_classes = 3 # change to actual number of classes
                cam = []
                for class_idx in range(num_classes):
                    self.model.zero_grad(set_to_none=True) # Clear gradients for all parameters 
                    # Clear stored gradients for attention maps
                    if hasattr(self.model, "attention_maps"):
                        for attn in self.model.attention_maps:
                            if hasattr(attn, "grad") and attn.grad is not None:
                                attn.grad = None
                    
                    logits = self.model(batch["source"], save_attn=True) # Forward pass with attention storage
                    class_specific_logits = logits[:, class_idx]
                    for attn in self.model.attention_maps:
                        attn.retain_grad() # command to retain gradients for every transformer layer
                    
                    class_specific_logits.backward()  # Compute gradients for each class
                    patch_attn_maps = self.model.attention_maps # list of [B*D, Heads, Tokens, Tokens]
                    patch_cam = self._grad_sam(patch_attn_maps, num_special=self.num_special,class_specific = False)
                    cam.append(patch_cam)
                    del logits, class_specific_logits # Free memory
                cam = torch.stack(cam, dim=0).mean(dim=0) # Average over classes
                cls_attn = cam

        elif self.attention_method in ["grad_rollout", "nonclass_grad_rollout"]:
            if self.attention_method == "grad_rollout":
                if target_class is None:
                    target_class = logits.argmax(dim=-1)
                score = logits[:, target_class]
    
                for attn in self.model.attention_maps:
                    attn.retain_grad() # command to retain gradients for every transformer layer
                self.model.zero_grad() # clear all gradient before backward pass
                score.backward()

                cls_attn = self._grad_rollout(
                    self.model.attention_maps, num_special=self.num_special
                )
            elif self.attention_method == "nonclass_grad_rollout":
                num_classes = 3 # change to actual number of classes
                rollout = []
                for class_idx in range(num_classes):
                    self.model.zero_grad(set_to_none=True) # Clear gradients for all parameters 
                    # Clear stored gradients for attention maps
                    if hasattr(self.model, "attention_maps"):
                        for attn in self.model.attention_maps:
                            if hasattr(attn, "grad") and attn.grad is not None:
                                attn.grad = None
                    
                    logits = self.model(batch["source"], save_attn=True) # Forward pass with attention storage
                    class_specific_logits = logits[:, class_idx]
                    for attn in self.model.attention_maps:
                        attn.retain_grad() # command to retain gradients for every transformer layer
                    
                    class_specific_logits.backward()  # Compute gradients for each class
                    rollout.append(self._grad_rollout(
                        self.model.attention_maps, num_special=self.num_special, class_specific=False
                    ))
                    del logits, class_specific_logits # Free memory
                cls_attn = torch.stack(rollout, dim=0).mean(dim=0) # Average over classes
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

    def _attention_rollout(self, attn_maps, num_special):

        rollout = None

        for attn in attn_maps:
            # [B, Heads, N , N]
            # attn = attn.max(dim=1).values  # Max over heads to get [B, N, N]
            attn = attn.mean(dim=1)  # Average over heads to get [B, N, N]
            # Discard lowes attention values (reduce noise)
            B, N, _ = attn.shape

            # if discard_ratio > 0:
                # k = int(N * discard_ratio)
                # if k > 0:
                    ## find indices of smallest values per row
                    # vals, idx = torch.topk(attn, k=k, dim=-1, largest=False)
                    ## build mask
                    # mask = torch.zeros_like(attn, dtype=torch.bool)
                    # mask.scatter_(-1, idx, True)
                    ## protect CLS + special tokens (columns)
                    # mask[:, :, :1 + num_special] = False
                    ## apply mask
                    # attn = attn.masked_fill(mask, 0)

            I = torch.eye(attn.size(-1), device=attn.device) # [N, N]
            
            I = I.unsqueeze(0) # [1, N, N]

            attn = attn + I # Add residual connection # [B, N, N]
            attn = attn / attn.sum(dim=-1, keepdim=True) # Normalize to make sum = 1 # [B, N, N]

            rollout = attn if rollout is None else torch.matmul(attn, rollout) # Recursive multiplication # [B, N, N] @ [B, N, N] → [B, N, N]
        # CLS → patches

        rollout = rollout[:, 0, 1+num_special:]   # [B, N]

        rollout = rollout / rollout.sum(dim=-1, keepdim=True) # [B, N] Normalize final rollout

        return rollout
    
    def _grad_sam(self, attn_maps, num_special,class_specific = True):
        cams = []
        for attn in attn_maps:
            grad = attn.grad
            if grad is None:
                continue
            # Grad-SAM
            if class_specific == True:
                cam = attn * torch.relu(grad) # [32, 12, 201, 201]
            elif class_specific == False:
                cam = attn * (grad.abs()) # [32, 12, 201, 201] Keep both positive and negative contributions
            # aggregate heads
            cam = cam.mean(dim=1) # [32, 201, 201]
            # aggregrate on j dimension (tokens attended to)
            cam = cam.mean(dim=-1) # [32, 201] 

            cams.append(cam)

        if len(cams) == 0:
            raise RuntimeError("No Grad-SAM gradients found.")

        # aggregate layers
        cam = torch.stack(cams).mean(dim=0) # [12, 32, 201] -> [32, 201]

        # remove special tokens (CLS + extra tokens)
        cam = cam[:, 1 + num_special:] # [32, 196]

        # # normalize
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8) 

        return cam 

    def _grad_rollout(self, attn_maps, num_special, class_specific = True,discard_ratio=0.9):
        rollout = None

        for attn in attn_maps:
            grad = attn.grad
            if grad is None:
                continue
            # # Grad Rollout
            # cam = attn * torch.relu(grad) # [32, 12, 201, 201]
            # # aggregate heads
            # cam = cam.mean(dim=1) # [32, 201, 201]

            cam = (attn * grad).mean(dim=1) # [32, 201, 201] Average over heads
            if class_specific:
                cam = torch.relu(cam) # ReLU to keep only positive contributions
            else:
                cam = cam.abs() # Keep both positive and negative contributions

            B, N, _ = cam.shape # [B = 32, N = 201]

            if discard_ratio > 0:
                flat = cam.view(cam.size(0), -1) 
                k = int(flat.size(-1) * discard_ratio)
                _, indices = flat.topk( 
                    k=k, 
                    dim=-1, 
                    largest=False) # Indices of lowest values
                batch_indices = torch.arange(
                    B, device=flat.device).unsqueeze(-1) 
                flat[batch_indices, indices] = 0 # Zero out lowest values # Apply independently to every slice
                cam = flat.view_as(cam) # Reshape back to [B, N, N
            I = torch.eye(cam.size(-1), device=cam.device).unsqueeze(0) # [1, N, N]
            cam = cam + I # Add residual connection # [B, N, N]
            cam = cam / cam.sum(dim=-1, keepdim=True) # Normalize to make sum = 1 # [B, N, N]

            rollout = cam if rollout is None else torch.matmul(cam, rollout) # Recursive multiplication # [B, N, N] @ [B, N, N] → [B, N, N]

        if rollout is None:
            raise RuntimeError("No Grad Rollout gradients found.")

        rollout = rollout[:, 0, 1 + num_special:]   # CLS → patches

        rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8) # Normalize final rollout

        return rollout
    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor):
        x = x - x.min()
        x = x / (x.max() - x.min() + 1e-8)
        return x