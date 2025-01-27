from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from . import register_vision_tower
from .base import VisionTower
from .dinov2_head import DistillDINOv2
from .mask2former_head import DistillMaskFormer
from .efficient_head import EfficientHead, CustomModelConfig

ckpt = "/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained/pytorch_model.bin"

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

# class EfficientHead(nn.Module):

#     def __init__(self):
#         super(EfficientHead, self).__init__()
        
#         self.dinov2_head = DistillDINOv2()
#         self.mask2former_head = DistillMaskFormer()
#         state_dict = torch.load(ckpt)
#         print(f"EfficientHead load ckpt: {self.load_state_dict(state_dict)}")

#     @torch.no_grad()
#     def forward(self, x):
#         self.eval()
#         decoded_features = self.dinov2_head(x)
#         topk_mask_queries, topk_labels = self.mask2former_head(x)
#         return decoded_features, topk_mask_queries, topk_labels


class CrossModalAttention(nn.Module):
    def __init__(self, config=None):
        super(CrossModalAttention, self).__init__()

        self.embed_dim = 1152
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
    
    
    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)
        
        # config = CustomModelConfig.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained")
        # self.efficient_head = EfficientHead(config)
        self.efficient_head = EfficientHead.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained")
        self.efficient_head.requires_grad_(False)
        
        self.text_projection = nn.Linear(2048, 1152) # qwen0.5b 896, qwen2.5b 2048
        self.query_projection = nn.Linear(256, 1152)
        self.dinov2_projection = nn.Linear(1024, 1152)
        self.fusion_hints = CrossModalAttention()
        
        self.query_projection.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.dinov2_projection.requires_grad_(True)
        self.fusion_hints.requires_grad_(True)
        
        self.class_embeds = nn.Embedding(20, 256)
        self.class_embeds.requires_grad_(True)
        self.has_class = True
        self.num_k = 16
        self.frames = 5

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            vision_tower_name = "/mnt/csi-data-aly/shared/public/haozhou/checkpoints/siglip/siglip-so400m-patch14-384/"
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                import os
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        
    
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
        # import pdb; pdb.set_trace()
        prompt_image_features, object_queries, topk_labels = self.efficient_head(ori_siglip_features)

        if self.has_class:
            topk_labels_embeds = self.class_embeds(topk_labels)
            object_queries += topk_labels_embeds
        
        text_embedding = self.text_projection(inputs_embeds.to(dtype=self.dtype)) # B, N, D
        queries_embedding = self.query_projection(object_queries.to(dtype=self.dtype)) # B, N, D
        prompt_image_features = self.dinov2_projection(prompt_image_features)

        bs, seq_len, dim = image_features.shape
        image_features = image_features.reshape(bs//self.frames, self.frames, seq_len, dim)
        prompt_image_features = prompt_image_features.reshape(bs//self.frames, self.frames, seq_len, dim)
        queries_embedding = queries_embedding.reshape(bs//self.frames, self.frames, queries_embedding.size(1), dim)
        text_embedding = text_embedding.reshape(bs//self.frames, self.frames, text_embedding.size(1), dim)
        
        query = image_features[:, -1, :, :]
        key_value = torch.cat([image_features.reshape(bs//self.frames, self.frames*seq_len, dim), 
                               prompt_image_features[:, -1, :, :], 
                               queries_embedding[:, -1, :, :], 
                               text_embedding[:, -1, :, :]], 
                               dim=1)
        
        image_features = query + self.fusion_hints(query, key_value)
        return image_features.to(x.dtype)
