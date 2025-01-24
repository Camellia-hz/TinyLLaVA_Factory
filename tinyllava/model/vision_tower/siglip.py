from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_vision_tower
from .base import VisionTower
from .efficient_head import EfficientHead


class CrossModalAttention(nn.Module):
    def __init__(self, config=None):
        super(CrossModalAttention, self).__init__()

        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = 0.1
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(self.embed_dim)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, image_features, text_embedding):
        B, T, C = image_features.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = int(C)
        T = int(T)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(image_features)
        k = self.k_proj(text_embedding)
        v = self.v_proj(text_embedding)
        if self.flash:
            k = k.view(B, k.size(1), self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = k.view(B, k.size(1), self.num_heads, C // self.num_heads).permute((0, 2, 3, 1)) # (B, nh, hs, T)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, v.size(1), self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Full attention set is_causal as False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask=None, 
                                                                 dropout_p=self.dropout if self.training else 0., 
                                                                 is_causal=False)
        else:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k) * (1.0 / math.sqrt(q.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


@register_vision_tower('siglip')      
class SIGLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = SiglipVisionModel(cfg)
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
        
        self.efficient_head = EfficientHead.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained")
        self.efficient_head.requires_grad_(False)
        
        self.text_projection = nn.Linear(896, 1024) # qwen0.5b 896, 
        self.query_projection = nn.Linear(256, 1024)
        self.siglip_feature_projection = nn.Linear(1152, 1024)
        self.fusion_hints = CrossModalAttention()
        
        self.query_projection.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.siglip_feature_projection.requires_grad_(True)
        self.fusion_hints.requires_grad_(True)
        
        self.class_embeds = nn.Embedding(20, 256)
        self.class_embeds.requires_grad_(True)
        self.has_class = True
        self.num_k = 16
        # if self.has_class and model_path is not None and os.path.exists(os.path.join(model_path, "model-00003-of-00003.safetensors")):
        #     from safetensors.torch import load_file
        #     state_dict = load_file(os.path.join(model_path, "model-00003-of-00003.safetensors"))
        #     state_dict_real = {
        #         k.replace('model.vision_tower.', ''): v
        #         for k, v in state_dict.items()
        #     }
        #     missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
    @property
    def dtype(self):
        return self._vision_tower.dtype
    
    def forward(self, x, inputs_embeds=None, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        ori_siglip_features = image_features.clone()
        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        prompt_image_features, object_queries, topk_labels = self.efficient_head(ori_siglip_features)

        if self.has_class:
            topk_labels_embeds = self.class_embeds(topk_labels)
            object_queries += topk_labels_embeds
        
        text_embedding = self.text_projection(inputs_embeds.to(dtype=self.dtype)) # B, N, D
        queries_embedding = self.query_projection(object_queries.to(dtype=self.dtype)) # B, N, D
        image_features = self.siglip_feature_projection(image_features)
        
        prompt_features = torch.cat([image_features, prompt_image_features, queries_embedding, text_embedding], dim=1)
        image_features = image_features.to(dtype=self.dtype) + self.fusion_hints(image_features.to(dtype=self.dtype), prompt_features)
        return image_features.to(x.dtype)
