
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)  
        
        if self.attention_mode == 'xformers': 
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class TransformerBlock(nn.Module):
    """
    A ViT3D tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of ViT3D.
    """
    def __init__(self, hidden_size, num_class, num_frame):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.num_frame = num_frame

    def forward(self, x):
        x = self.linear(self.norm_final(rearrange(x, '(b f) d -> b (f d)', f=self.num_frame)))
        return x
    
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

class ViT3D(nn.Module):

    def __init__(
        self,
        input_size=256,
        patch_size=8,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=2,
        learn_sigma=True,
        attention_mode='math',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.hidden_size =  hidden_size
        # self.pooling = nn.AdaptiveAvgPool1d(64)
        # self.linear = nn.Linear(in_features=384, out_features=1152)
        # self.linear_2 = nn.Linear(in_features=384*4, out_features=1152)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size*self.num_frames, self.num_classes, self.num_frames)


        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Zero-out adaLN modulation layers in ViT3D blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, x, attentions=None, use_fp16=False):
        
        """
        Forward pass of ViT3D.
        x: (N, F, C, H, W) tensor of video inputs
        y: (N,) tensor of class labels
        """
        # if use_fp16:
        #     x = x.to(dtype=torch.float16)
        batches, slices, channels, high, weight = x.shape 

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x)

        cls_tokens = self.cls_token.expand(batches*slices, -1, -1) 
        
        x = x + self.pos_embed            
        x = torch.cat((cls_tokens, x), dim=1)     

        #torch.Size([48, 1025, 384])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            
            x = spatial_block(x)
            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed

            x = temp_block(x)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

        x = x[:, 0]
        x = self.final_layer(x)               
        
        return x



def ViT3D_XL_2(**kwargs):
    return ViT3D(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def ViT3D_XL_4(**kwargs):
    return ViT3D(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def ViT3D_XL_8(**kwargs):
    return ViT3D(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def ViT3D_L_2(**kwargs):
    return ViT3D(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def ViT3D_L_4(**kwargs):
    return ViT3D(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def ViT3D_L_8(**kwargs):
    return ViT3D(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def ViT3D_B_2(**kwargs):
    return ViT3D(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def ViT3D_B_4(**kwargs):
    return ViT3D(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def ViT3D_B_8(**kwargs):
    return ViT3D(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def ViT3D_S_2(**kwargs):
    return ViT3D(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def ViT3D_S_4(**kwargs):
    return ViT3D(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def ViT3D_S_8(**kwargs):
    return ViT3D(depth=22, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

ViT3D_models = {
    'ViT3D-XL/2': ViT3D_XL_2,  'ViT3D-XL/4': ViT3D_XL_4,  'ViT3D-XL/8': ViT3D_XL_8,
    'ViT3D-L/2':  ViT3D_L_2,   'ViT3D-L/4':  ViT3D_L_4,   'ViT3D-L/8':  ViT3D_L_8,
    'ViT3D-B/2':  ViT3D_B_2,   'ViT3D-B/4':  ViT3D_B_4,   'ViT3D-B/8':  ViT3D_B_8,
    'ViT3D-S/2':  ViT3D_S_2,   'ViT3D-S/4':  ViT3D_S_4,   'ViT3D-S/8':  ViT3D_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    img = torch.randn(3, 16, 3, 256, 256).to(device)
    network = ViT3D_S_8().to(device)

    y = network(img)
    print(y)
