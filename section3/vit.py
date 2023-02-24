"""
The source is ViT from lucidrains implementation
https://github.com/lucidrains/vit-pytorch
"""


import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from torch.profiler import record_function

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # FIX: nn.BatchNorm1d(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # (づ ◕‿◕ )づ
        with record_function(f"PRENORM"):
            # x = rearrange(x, "b l c -> b c l")
            x = self.norm(x)
            # x = rearrange(x, "b c l -> b l c")
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # ╭( -᷅_-᷄ 〝)╮
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.ReLU(), # FIX: GELU()
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        with record_function(f"FEEDFORWARD"):
            x = self.net[0](x)
            with record_function("RELU"): # можно было и без этого, но так удобнее искать в таблице
                x = self.net[1](x)
            for i in range(2, len(self.net)):
                x = self.net[i](x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # ˁ ुᴗ_ᴗ)ु.｡oO
        self.lin = nn.Linear(dim, 3*inner_dim, bias=False) # FIX: 1 linear instead of 3
        self.inner_dim = inner_dim

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        with record_function(f"ATTENTION"):
            q, k, v = torch.split(self.lin(x), self.inner_dim, dim=-1)
            # FIX: Forgot head dimension (effectively had 1 head instead of 8)
            b, l, inner = q.shape
            q = q.view(b, l, self.heads, -1)
            k = k.view(b, l, self.heads, -1)
            v = v.view(b, l, self.heads, -1)
            # 〈╭☞• ⍛•〉╭☞
            dots = torch.matmul(q, k.transpose(-1, -3)) * self.scale
            dots = dots.transpose(-1, -2)
            with record_function("SOFTMAX"):
                attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        # ヾ( • – •*)〴

        with record_function("POS INIT"):
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        with record_function("PATCH"):
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        with record_function("POS ADD"):
            x += self.pos_embedding[:, : (n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        with record_function("MLP HEAD"):
            output = self.mlp_head(x)
        return output
