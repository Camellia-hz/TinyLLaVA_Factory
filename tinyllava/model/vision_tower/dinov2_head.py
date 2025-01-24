import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

transformer_kwargs = {
    "dim": 512,
    "depth": 4,
    "heads": 8,
    "dim_head": 128,
    "mlp_dim": 512 * 4, 
    "dropout": 0.1
}

CKPT = "/mnt/csi-data-aly/shared/public/openpilot_deepdive/haozhou/image_tokenizer_exp/runs/siglip_dinov2_head/epoch_5.pth"

class DistillDINOv2(nn.Module):
    def __init__(self, ckpt=CKPT):
        super(DistillDINOv2, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(1152, 512),
            Transformer(**transformer_kwargs),
            nn.Linear(512, 1024)
        )
        # state_dict = torch.load(ckpt, map_location='cpu')
        # state_dict_real = {
        #     k.replace('module.', ''): v
        #     for k, v in state_dict.items()
        # }
        # missing, unexpected = self.load_state_dict(state_dict_real, strict=False)
        # assert len(missing) == 0
        self.decoder.requires_grad_(False)
    
    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype

    @property
    def device(self):
        return next(self.decoder.parameters()).device
    
    @torch.no_grad()
    def forward(self, clip_features):
        self.decoder.eval()
        decoded_features = self.decoder(clip_features.to(device=self.device, dtype=self.dtype))
        return decoded_features[:, 1:]


if __name__ == "__main__":
    model = DistillDINOv2().cuda().to(dtype=torch.float16)
    x = torch.randn(2, 729, 1152)
    out = model(x)
    print(out.shape)
    print(out.dtype)