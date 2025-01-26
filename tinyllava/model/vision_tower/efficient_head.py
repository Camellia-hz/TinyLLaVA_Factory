import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from tinyllava.model.vision_tower.dinov2_head import DistillDINOv2
from tinyllava.model.vision_tower.mask2former_head import DistillMaskFormer


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_model_with_two_heads"
    
    def __init__(self, dinov2_dim=512, mask2former_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.dinov2_dim = dinov2_dim
        self.mask2former_dim = mask2former_dim


class EfficientHead(PreTrainedModel):
    config_class = CustomModelConfig

    def __init__(self, config=None):
        super(EfficientHead, self).__init__(config=config)
        
        self.dinov2_head = DistillDINOv2()
        self.mask2former_head = DistillMaskFormer()
        self.post_init()

    @torch.no_grad()
    def forward(self, x):
        self.eval()
        # import pdb; pdb.set_trace()
        ori_dtype = x.dtype
        self.to(dtype=torch.float32)
        decoded_features = self.dinov2_head(x.float()).to(dtype=ori_dtype)
        topk_mask_queries, topk_labels = self.mask2former_head(x.float())
        return decoded_features, topk_mask_queries.to(dtype=ori_dtype), topk_labels


if __name__ == "__main__":
    config = CustomModelConfig.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained")
    # model = EfficientHead(config)
    model = EfficientHead.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/pretrained")
    print(model.requires_grad_(False))
    # 示例输入
    x = torch.randn(2, 729, 1152)
    output = model(x)
    print(output)