import torch 
from .base_model import BasicClassifier
# from transformers import Dinov2Model
from .utils.transformer_blocks import TransformerEncoderLayer
import torch.nn as nn
from einops import rearrange
from .extern.dinov2.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

def slices2rgb(tensor):
    # [B, 1, D, H, W] -> [B*D//3, 3, H, W]
    B, C, D, H, W = tensor.shape

    assert C == 1, "More than one channel"

    # If D is not divisible by 3, we need to pad by repeating the first slice
    if D % 3 != 0:
        padding_size = 3 - (D % 3)  # Find out how much padding is needed
        padding = tensor[:, :, :padding_size]  # Take the first slices to pad
        tensor = torch.cat([tensor, padding], dim=2)  # Concatenate along D axis
    
    # Reshape the tensor from [B, 1, D, H, W] to [B * (D // 3), 3, H, W]
    B, _, D, H, W = tensor.shape
    tensor = tensor.view(B, D // 3, 3, H, W)  # Reshape to [B, D//3, 3, H, W]
    tensor = tensor.reshape(-1, 3, H, W)  # [B*D//3, 3, H, W]
    
    return tensor 

   


class DinoV2ClassifierSlice(BasicClassifier):
    def __init__(
            self, 
            in_ch,
            out_ch,
            spatial_dims=2,
            pretrained=True,
            save_attn = False,
            rotary_positional_encoding=None,
            optimizer_kwargs={'lr': 1e-6, 'weight_decay': 1e-2},
            model_size = 's', # [s, b, l, 'g']
            use_registers = False,
            use_bottleneck=False,
            use_slice_pos_emb=False,
            enable_linear = True,
            enable_trans = True, # Deprecated 
            slice_fusion='transformer',
            freeze=False,
            dino='v2',
            attn_mask=None,
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims=spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.save_attn = save_attn
        self.attention_maps = []
        self.attention_maps_slice = []
        self.use_registers = use_registers
        self.slice_fusion_type = slice_fusion
        self.attn_mask = attn_mask

        if pretrained:
            if dino == 'v2':
                if use_registers:
                    self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14_reg')
                else:
                    self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            elif dino == 'v3-vit':
                if model_size == 's':
                    self.encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vits16plus',
                                                 weights=f"../dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                                                 trust_repo=True, skip_validation=True)
                elif model_size == 'b':
                    self.encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vitb16',
                                                     weights=f"../dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                                                     trust_repo=True, skip_validation=True)         
                else:
                    raise NotImplementedError()
        else:
            assert dino == 'v2'
            Model = {'s': vit_small, 'b': vit_base, 'l':vit_large, 'g':vit_giant2 }[model_size]
            self.encoder = Model(patch_size=14, num_register_tokens=0)
   
        # Freeze backbone 
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    
        emb_ch = self.encoder.num_features 
        if use_bottleneck:
            self.bottleneck = nn.Linear(emb_ch, emb_ch//4)
            emb_ch = emb_ch//4 
        self.emb_ch = emb_ch

        if slice_fusion == 'transformer':
            if use_slice_pos_emb:
                self.slice_pos_emb = nn.Embedding(256, emb_ch) # WARNING: Assuming max. 256 slices

            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=12 if emb_ch == 768 else 16,
                    dim_feedforward=1*emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    rotary_positional_encoding=rotary_positional_encoding
                ),
                num_layers=1,
                norm=nn.LayerNorm(emb_ch)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion == 'linear':
            emb_ch = emb_ch*32
        elif slice_fusion == 'average':
            pass 

        self.linear = nn.Linear(emb_ch, out_ch) if enable_linear else nn.Identity()



        


    def forward(self, source, save_attn=False, src_key_padding_mask=None, **kwargs):   
        if save_attn and self.attn_mask is not None:
            raise NotImplementedError("Attention mask not implemented for attention saving mode.")

        if save_attn:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()
        
        if self.attn_mask is not None:
            self.register_attention_mask(self.attn_mask)


        x = source.to(self.device) # [B, C, D, H, W]
        B, C, *_ = x.shape
 

        x = rearrange(x, 'b c d h w -> (b d c) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # x = slices2rgb(x) # [B, 1, D, H, W] -> [B*D//3, 3, H, W]

        x = self.encoder(x) # [(B D), C, H, W] -> [(B D), out] 

        # Bottleneck: force to focus on relevant features for classification 
        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)
        
        # Slice fusion 
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if hasattr(self, 'slice_pos_emb'):
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            x += self.slice_pos_emb(pos)
        
        if self.slice_fusion_type == 'transformer':
            x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)
 
            if src_key_padding_mask is not None: 
                src_key_padding_mask = src_key_padding_mask.to(self.device)
                src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                src_key_padding_mask = torch.concat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)# [Batch, L]
       
            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
            x = x[:, 0]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b (d e)')
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=False)

        if save_attn:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        if self.attn_mask is not None:
            self.deregister_attention_mask()

        # Logits 
        if kwargs.get('without_linear', False):
            return x 
        x = self.linear(x) 
        return x
    



    
    def get_slice_attention(self):
        attention_map_slice = self.attention_maps_slice[-1] # [B, Heads, 1+D(+regs), 1+D(+regs)]
        attention_map_slice = attention_map_slice[:, :, 0, 1:] # [B, Heads, D]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True)

        # Option 1:
        attention_map_slice = attention_map_slice.mean(dim=1)  # [B, D]
        attention_map_slice = attention_map_slice.view(-1) # [B*D]
        attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]

        # Option 2:
        # attention_map_slice = rearrange(attention_map_slice, 'b d e -> (b e) d') # [B*D, Heads]
        # attention_map_slice = attention_map_slice[:, :, None] # [B*D, Heads, 1]

        return attention_map_slice

    def get_plane_attention(self):
        attention_map_dino = self.attention_maps[-1] # [B*D, Heads, 1+HW, 1+HW]
        img_slice = slice(5, None) if self.use_registers else slice(1, None) # see https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L264 
        attention_map_dino = attention_map_dino[:,:, 0, img_slice] # [B*D, Heads, HW]
        attention_map_dino[:,:,0] = 0
        attention_map_dino /= attention_map_dino.sum(dim=-1, keepdim=True)
        return attention_map_dino

    def get_attention_maps(self):
        attention_map_dino = self.get_plane_attention()
        attention_map_slice = self.get_slice_attention()
        
        attention_map = attention_map_slice*attention_map_dino
        return attention_map
    
    def get_attention_cls(self):
        """ Calculate the attention in the first layer starting from the CLS token in the last layer. """
        attention_to_cls = self.attention_maps[-1]
        # Propagate the attention backwards
        for attn in reversed(self.attention_maps[:-1]):
            attention_to_cls = torch.matmul(attn, attention_to_cls)
        
        # The attention to the first layer from the CLS token
        return attention_to_cls
    
    def register_hooks(self):
        def enable_attention(module):
            forward_orig = module.forward
            def forward_wrap(*args, **kwargs): 
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            module.forward = forward_wrap
            module.foward_orig = forward_orig

        def enable_attention2(mod):
            forward_orig = mod.forward
            def forward_wrap(self2, x):
                # forward_orig.__self__
                B, N, C = x.shape
                qkv = self2.qkv(x).reshape(B, N, 3, self2.num_heads, C // self2.num_heads).permute(2, 0, 3, 1, 4)
                
                q, k, v = qkv[0] * self2.scale, qkv[1], qkv[2]
                attn = q @ k.transpose(-2, -1)
        
                attn = attn.softmax(dim=-1)
                attn = self2.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self2.proj(x)
                x = self2.proj_drop(x)

                # Hook attention map 
                self.attention_maps.append(attn)

                return x
            
            mod.forward = lambda x, *args, **kwargs: forward_wrap(mod, x, *args, **kwargs)
            mod.foward_orig = forward_orig

        def enable_attention3(mod):
            compute_attention_original = mod.compute_attention
            def compute_attention_wrap(self3, qkv, *args, **kwargs):
                
                attn_bias = kwargs.get('attn_bias', None)
                assert attn_bias is None

                rope = kwargs.get('rope', None)
                
                B, N, _ = qkv.shape
                C = self3.qkv.in_features

                qkv = qkv.reshape(B, N, 3, self3.num_heads, C // self3.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
                if rope is not None:
                    q, k = self3.apply_rope(q, k, rope)

                # get attn by implementing scaled dot product attention
                # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                attn = q @ k.transpose(-2, -1) * self3.scale
                attn = torch.softmax(attn, dim=-1)
                attn = self3.attn_drop(attn)
                x = attn @ v

                # Hook attention map 
                self.attention_maps.append(attn)

                x = x.transpose(1, 2)
                return x.reshape([B, N, C])
            mod.compute_attention = lambda qkv, *args, **kwargs: compute_attention_wrap(mod, qkv, *args, **kwargs)
            mod.compute_attention_original = compute_attention_original                


        def append_attention_maps(module, input, output):
            self.attention_maps_slice.append(output[1])

        # Hook Dino Attention
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                # DINOv3
                if hasattr(mod, 'compute_attention'):
                    enable_attention3(mod)
                # DINOv2
                else:
                    enable_attention2(mod)

        # Hook Slice Attention
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.hooks.append(mod.register_forward_hook(append_attention_maps))


    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()

        # Dino Attention
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                if hasattr(mod, 'compute_attention_original'):
                    mod.compute_attention = mod.compute_attention_original
                elif hasattr(mod, 'foward_orig'):
                     mod.forward = mod.foward_orig
   
        # Slice Attention
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = mod.foward_orig

    def register_attention_mask(self, attn_mask):
        def add_attention_mask_v3(mod):
            compute_attention_original = mod.compute_attention
            def compute_attention_injected(self4, qkv, *args, **kwargs):
                
                attn_bias = kwargs.get('attn_bias', None)
                assert attn_bias is None

                rope = kwargs.get('rope', None)
                
                B, N, _ = qkv.shape
                C = self4.qkv.in_features

                qkv = qkv.reshape(B, N, 3, self4.num_heads, C // self4.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
                if rope is not None:
                    q, k = self4.apply_rope(q, k, rope)

                if attn_mask is not None:
                    assert attn_mask.shape[-2:] == (N, N)
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

                x = x.transpose(1, 2)
                return x.reshape([B, N, C])
            mod.compute_attention = lambda qkv, *args, **kwargs: compute_attention_injected(mod, qkv, *args, **kwargs)
            mod.compute_attention_original = compute_attention_original

        # Hook Dino Attention
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                # DINOv3
                if hasattr(mod, 'compute_attention'):
                    add_attention_mask_v3(mod)                

    def deregister_attention_mask(self):
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                if hasattr(mod, 'compute_attention_original'):
                    mod.compute_attention = mod.compute_attention_original
