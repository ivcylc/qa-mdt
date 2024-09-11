from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch as th
#from .utils_pos_embedding.pos_embed import RoPE2D
import torch.nn as nn
import torch.nn.functional as F
import sys
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from timm.models.layers import DropPath
from .utils import auto_grad_checkpoint, to_2tuple
from .PixArt_blocks import t2i_modulate, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, FinalLayer
import xformers.ops

import math
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=(256, 16),
            patch_size=(16, 4),
            overlap = (0, 0),
            in_chans=128,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        
        # img_size=(256, 16)
        # patch_size=(16, 4)
        # overlap = (2, 2)
        # in_chans=128
        # embed_dim=768
        # import pdb
        # pdb.set_trace()
        self.img_size = img_size
        self.patch_size = patch_size
        self.ol = overlap
        self.grid_size = (math.ceil((img_size[0] - patch_size[0]) / (patch_size[0]-overlap[0])) + 1, 
                          math.ceil((img_size[1] - patch_size[1]) / (patch_size[1]-overlap[1])) + 1)
        self.pad_size = ((self.grid_size[0]-1) * (self.patch_size[0]-overlap[0])+self.patch_size[0]-self.img_size[0], 
                        +(self.grid_size[1]-1)*(self.patch_size[1]-overlap[1])+self.patch_size[1]-self.img_size[1])
        self.pad_size = (self.pad_size[0] // 2, self.pad_size[1] // 2)
        # self.p-ad_size = (((img_size[0] - patch_size[0]) // ), )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0]-overlap[0], patch_size[1]-overlap[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        
        
        x = F.pad(x, (self.pad_size[-1], self.pad_size[-1], self.pad_size[-2], self.pad_size[-2]), "constant", 0)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
class PatchEmbed_1D(nn.Module):
    def __init__(
            self,
            img_size=(256, 16),
            # patch_size=(16, 4),
            # overlap = (0, 0),
            in_chans=8,
            embed_dim=1152,
            norm_layer=None,
            # flatten=True,
            bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans*img_size[1], embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        
        # x = F.pad(x, (self.pad_size[-1], self.pad_size[-1], self.pad_size[-2], self.pad_size[-2]), "constant", 0)
        # x = self.proj(x)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = th.einsum('bctf->btfc', x)
        x = x.flatten(2)  # BTFC -> BTD
        x = self.proj(x)
        x = self.norm(x)
        return x

# if __name__ == '__main__':
#     x = th.rand(1, 256, 16).unsqueeze(0)
#     model = PatchEmbed(in_chans=1)
#     y = model(x)
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

from positional_encodings.torch_encodings import PositionalEncoding1D

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        # x [B, T, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
    
from audioldm_train.modules.diffusionmodules.attention import CrossAttention_1D
    
class PixArtBlock_Slow(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention_1D(query_dim=hidden_size, context_dim=hidden_size, heads=num_heads, dim_head=int(hidden_size/num_heads))
        self.cross_attn = CrossAttention_1D(query_dim=hidden_size, context_dim=hidden_size, heads=num_heads, dim_head=int(hidden_size/num_heads))
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        # x [B, T, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

class PixArt(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)
        drop_path = [x.item() for x in th.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # if config:
        #     logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        #     logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        # else:
        #     print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, context_list, context_mask_list=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        mask = context_mask_list[0] # (N, L)

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        
        # torch_map = th.zeros(self.x_embedder.img_size[0]+2*self.x_embedder.pad_size[0],
        #                       self.x_embedder.img_size[1]+2*self.x_embedder.pad_size[1]).to(x.device)
        # lf = self.x_embedder.grid_size[0]
        # rf = self.x_embedder.grid_size[1]

        # for i in range(lf):
        #     for j in range(rf):
                
        #         xx = (i) * (self.x_embedder.patch_size[0]-self.x_embedder.ol[0])
        #         yy = (j) * (self.x_embedder.patch_size[1]-self.x_embedder.ol[1])

        #         torch_map[xx:(xx+self.x_embedder.patch_size[0]), yy:(yy+self.x_embedder.patch_size[1])]+=1
        # torch_map = torch_map[self.x_embedder.pad_size[0]:self.x_embedder.pad_size[0]+self.x_embedder.img_size[0], 
        #                     self.x_embedder.pad_size[1]:self.x_embedder.pad_size[1]+self.x_embedder.img_size[1]]
        # torch_map = th.reciprocal(torch_map)
        # c = self.out_channels
        # p0, p1 = self.x_embedder.patch_size[0], self.x_embedder.patch_size[1]

        # x = x.reshape(shape=(x.shape[0], self.x_embedder.grid_size[0],
        #                       self.x_embedder.grid_size[1], p0, p1, c))
        # x = th.einsum('nhwpqc->nchwpq', x)
        # added_map = th.zeros(x.shape[0], c, 
        #                      self.x_embedder.img_size[0]+2*self.x_embedder.pad_size[0],
        #                      self.x_embedder.img_size[1]+2*self.x_embedder.pad_size[1]).to(x.device)
        
        # for b_id in range(x.shape[0]):
        #     for i in range(lf):
        #         for j in range(rf):
        #             for c_id in range(c):
        #                 xx = (i) * (self.x_embedder.patch_size[0]-self.x_embedder.ol[0])
        #                 yy = (j) * (self.x_embedder.patch_size[1]-self.x_embedder.ol[1])
        #                 added_map[b_id][c_id][xx:(xx+self.x_embedder.patch_size[0]), yy:(yy+self.x_embedder.patch_size[1])] += \
        #                 x[b_id, c_id, i, j]
        # ret_map = th.zeros(x.shape[0], c, self.x_embedder.img_size[0],
        #                       self.x_embedder.img_size[1]).to(x.device)
        # for b_id in range(x.shape[0]):
        #     for id_c in range(c):
        #         ret_map[b_id, id_c, :, :] = th.mul(added_map[b_id][id_c][self.x_embedder.pad_size[0]:self.x_embedder.pad_size[0]+self.x_embedder.img_size[0], 
        #                             self.x_embedder.pad_size[1]:self.x_embedder.pad_size[1]+self.x_embedder.img_size[1]], torch_map)
        c = self.out_channels
        p0 = self.x_embedder.patch_size[0]
        p1 = self.x_embedder.patch_size[1]
        h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # h = w = int(x.shape[1] ** 0.5)
        # print(x.shape, h, w, p0, p1)
        # import pdb
        # pdb.set_trace()

        x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
        x = th.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
class DEBlock(nn.Module):
    """
    Decoder block with added SpecTNT transformer
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, FFN_type='SwiGLU', drop_path=0., window_size=0, input_size=None, use_rel_pos=False, skip=False, num_f=None, num_t=None, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        # self.cross_attn_f = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        # self.cross_attn_t = MultiHeadCrossAttention(hidden_size*num_f, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm5 = nn.LayerNorm(hidden_size * num_f, elementwise_affine=False, eps=1e-6)
        self.norm6 = nn.LayerNorm(hidden_size * num_f, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if FFN_type == 'mlp':
            self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
            # self.mlp2 = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
            # self.mlp3 = Mlp(in_features=hidden_size*num_f, hidden_features=int(hidden_size*num_f * mlp_ratio), act_layer=approx_gelu, drop=0)
        elif FFN_type == 'SwiGLU':
            self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio), 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.scale_shift_table_2 = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.scale_shift_table_3 = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

        self.F_transformer = WindowAttention(hidden_size, num_heads=4, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        
        self.T_transformer = WindowAttention(hidden_size * num_f, num_heads=16, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)

        self.f_pos = nn.Embedding(num_f, hidden_size)
        self.t_pos = nn.Embedding(num_t, hidden_size * num_f)
        self.num_f = num_f 
        self.num_t = num_t

    def forward(self, x_normal, end, y, t, mask=None, skip=None, ids_keep=None, **kwargs):
        # import pdb 
        # pdb.set_trace()
        B, D, C = x_normal.shape
        T = self.num_t
        F_add_1 = self.num_f
        # B, T, F_add_1, C = x.shape
        # F_add_1 = F_add_1 + 1
        # x_normal = th.reshape()
        # # x_end [B, T, 1, C]
        # x_end = x[:, :, -1, :].unsqueeze(2)
        if self.skip_linear is not None:
            x_normal = self.skip_linear(th.cat([x_normal, skip], dim=-1))
        
        D = T * (F_add_1 - 1)
        # x_normal [B, D, C]
        # import pdb 
        # pdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x_normal = x_normal + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x_normal), shift_msa, scale_msa)).reshape(B, D, C))

        x_normal = x_normal.reshape(B, T, F_add_1-1, C)
        x_normal = th.cat((x_normal, end), 2)

        # x_normal [B*T, F+1, C]
        x_normal = x_normal.reshape(B*T, F_add_1, C)
        pos_f = th.arange(self.num_f, device=x.device).unsqueeze(0).expand(B*T, -1)
        # import pdb; pdb.set_trace()
        x_normal = x_normal + self.f_pos(pos_f)
        
        x_normal = x_normal + self.F_transformer(self.norm3(x_normal))
        # x_normal = x_normal + self.cross_attn_f(x_normal, y, mask)
        # x_normal = x_normal + self.mlp2(self.norm4(x_normal))

        # x_normal [B, T, (F+1) * C]
        x_normal = x_normal.reshape(B, T, F_add_1 * C)
        pos_t = th.arange(self.num_t, device=x.device).unsqueeze(0).expand(B, -1)
        x_normal = x_normal + self.t_pos(pos_t)
        x_normal = x_normal + self.T_transformer(self.norm5(x_normal))
        # x_normal = x_normal + self.cross_attn_t(x_normal, y, mask)


        x_normal = x_normal.reshape(B, T ,F_add_1, C)
        end = x_normal[:, :, -1, :].unsqueeze(2)
        x_normal = x_normal[:, :, :-1, :]
        x_normal = x_normal.reshape(B, T*(F_add_1 - 1), C)

        x_normal = x_normal + self.cross_attn(x_normal, y, mask)
        x_normal = x_normal + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x_normal), shift_mlp, scale_mlp)))
        
        # x_normal = th.cat
        return x_normal, end #.reshape(B, )
class MDTBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, FFN_type='mlp', drop_path=0., window_size=0, input_size=None, use_rel_pos=False, skip=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if FFN_type == 'mlp':
            self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        elif FFN_type == 'SwiGLU':
            self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio), 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)

        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, y, t, mask=None, skip=None, ids_keep=None, **kwargs):
        B, N, C = x.shape
        if self.skip_linear is not None:
            x = self.skip_linear(th.cat([x, skip], dim=-1))
        # x [B, T, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
class PixArt_MDT_MASK_TF(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=False, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, mask_t=0.17, mask_f=0.15, decode_layer=4,**kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth

        drop_path_half = [x.item() for x in th.linspace(0, drop_path, half_depth)]  # stochastic depth decay rule
        drop_path_decode = [x.item() for x in th.linspace(0, drop_path, decode_layer)]
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp') for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_decode[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp') for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for _ in range(1)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(th.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        # if mask_ratio is not None:
        #     self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
        #     self.mask_ratio = float(mask_ratio)
        #     self.decode_layer = int(decode_layer)
        # else:
        #     self.mask_token = nn.Parameter(th.zeros(
        #         1, 1, hidden_size), requires_grad=False)
        #     self.mask_ratio = None
        #     self.decode_layer = int(decode_layer)
        assert mask_t != 0 and mask_f != 0
        self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
        self.mask_t = mask_t 
        self.mask_f = mask_f 
        self.decode_layer = int(decode_layer)
        print(f"mask ratio: T-{self.mask_t} F-{self.mask_f}", "decode_layer:", self.decode_layer)
        self.initialize_weights()


    def forward(self, x, timestep, context_list, context_mask_list=None, enable_mask=False, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # import pdb 
        # pdb.set_trace()
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        # if not self.training:
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)
            print("MASK !!!!!!!!!!!!!!!!!!!!!!!!!")

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]

        input_skip = x

        masked_stage = False
        skips = []
        # TODO : masking op for training
        if self.mask_t is not None and self.training:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = th.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio_t = rand_mask_ratio * 0.13 + self.mask_t # mask_ratio, mask_ratio + 0.2 
            rand_mask_ratio_f = rand_mask_ratio * 0.13 + self.mask_f
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking_2d(
                x, rand_mask_ratio_t, rand_mask_ratio_f)
            masked_stage = True


        for block in self.en_inblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=None)

        if self.mask_t is not None and self.mask_f is not None and self.training:
            x = self.forward_side_interpolater(x, y, t0, y_lens, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, context_list, context_mask_list=None, cfg_scale=4.0, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # import pdb
        # pdb.set_trace()
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, context_list, context_mask_list=None)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        if self.x_embedder.ol == (0, 0) or self.x_embedder.ol == [0, 0]:
            c = self.out_channels
            p0 = self.x_embedder.patch_size[0]
            p1 = self.x_embedder.patch_size[1]
            h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

            x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
            x = th.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
            return imgs
    
        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]
        bs = x.shape[0]
        
        torch_map = self.torch_map

        c = self.out_channels

        x = x.reshape(shape=(bs, lf, rf, lp, rp, c))
        x = th.einsum('nhwpqc->nchwpq', x)

        added_map = th.zeros(bs, c, lm+2*lpad, rm+2*rpad).to(x.device)
        
        for i in range(lf):
            for j in range(rf):
                    xx = (i) * (lp - lo)
                    yy = (j) * (rp - ro)
                    added_map[:, :, xx:(xx+lp), yy:(yy+rp)] += \
                    x[:, :, i, j, :, :]
        # import pdb 
        # pdb.set_trace()
        added_map = added_map[:][:][lpad:lm+lpad, rpad:rm+rpad]
        return th.mul(added_map.to(x.device), torch_map.to(x.device))

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # if self.use_custom_patch: # overlapped patch
        #     T=101
        #     F=12
        # else:            
        #     T=64
        #     F=8
        T = self.x_embedder.grid_size[0]
        F = self.x_embedder.grid_size[1]
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = th.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = th.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = th.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = th.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = th.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = th.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = th.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = th.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = th.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = th.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=th.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = th.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = th.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = th.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore, ids_keep

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = th.argsort(noise, dim=1)
        ids_restore = th.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = th.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = th.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = th.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, y, t0, y_lens, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = th.cat([x, mask_tokens], dim=1)
        x = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, y, t0, y_lens, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.en_inblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.en_outblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.x_embedder.ol == [0, 0] or self.x_embedder.ol == (0, 0):
            return

        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]

        torch_map = th.zeros(lm+2*lpad, rm+2*rpad).to('cuda')
        for i in range(lf):
            for j in range(rf):
                xx = (i) * (lp - lo)
                yy = (j) * (rp - ro)
                torch_map[xx:(xx+lp), yy:(yy+rp)]+=1 
        torch_map = torch_map[lpad:lm+lpad, rpad:rm+rpad]
        self.torch_map = th.reciprocal(torch_map)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class PixArt_MDT_MOS_AS_TOKEN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=False, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, mask_ratio=None, decode_layer=4,**kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        decode_layer = int(decode_layer)

        self.mos_embed = nn.Embedding(num_embeddings=5, embedding_dim=hidden_size)        

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # self.mos_block = nn.Sequential(

        # )

        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth

        drop_path_half = [x.item() for x in th.linspace(0, drop_path, half_depth)]  # stochastic depth decay rule
        drop_path_decode = [x.item() for x in th.linspace(0, drop_path, decode_layer)]
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp') for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_decode[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp') for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for _ in range(1)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(th.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(th.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.initialize_weights()


    def forward(self, x, timestep, context_list, context_mask_list=None, enable_mask=False, mos=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # mos = th.ones(x.shape[0], dtype=th.int).to(x.device)
        #print(f'DEBUG! {x}, {mos}') 
        assert mos.shape[0] == x.shape[0]
        #import pdb; pdb.set_trace()
        mos = mos - 1
        mos = self.mos_embed(mos.to(x.device).to(th.int)) # [N, dim]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # import pdb 
        # pdb.set_trace()
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        # if not self.training:
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]

        masked_stage = False
        skips = []
        # TODO : masking op for training
        try:
            x = th.cat([mos, x], dim=1)  # [N, L+1, dim]
        except:
            x = th.cat([mos.unsqueeze(1), x], dim=1)
        input_skip = x

        if self.mask_ratio is not None and self.training:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = th.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True
        for block in self.en_inblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and self.training:
            x = self.forward_side_interpolater(x, y, t0, y_lens, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed
            # x = x + self.decoder_pos_embed
        # import pdb 
        # pdb.set_trace()
        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=this_skip, ids_keep=None)
        x = x[:, 1:, :]
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        # import pdb 
        # pdb.set_trace()
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, context_list, context_mask_list=None, cfg_scale=4.0, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # import pdb
        # pdb.set_trace()
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, context_list, context_mask_list=None)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        if self.x_embedder.ol == (0, 0) or self.x_embedder.ol == [0, 0]:
            c = self.out_channels
            p0 = self.x_embedder.patch_size[0]
            p1 = self.x_embedder.patch_size[1]
            h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

            x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
            x = th.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
            return imgs
    
        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]
        bs = x.shape[0]
        
        torch_map = self.torch_map

        c = self.out_channels

        x = x.reshape(shape=(bs, lf, rf, lp, rp, c))
        x = th.einsum('nhwpqc->nchwpq', x)

        added_map = th.zeros(bs, c, lm+2*lpad, rm+2*rpad).to(x.device)
        
        for i in range(lf):
            for j in range(rf):
                    xx = (i) * (lp - lo)
                    yy = (j) * (rp - ro)
                    added_map[:, :, xx:(xx+lp), yy:(yy+rp)] += \
                    x[:, :, i, j, :, :]
        # import pdb 
        # pdb.set_trace()
        added_map = added_map[:][:][lpad:lm+lpad, rpad:rm+rpad]
        return th.mul(added_map.to(x.device), torch_map.to(x.device))

    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        L = L - 1
        len_keep = int(L * (1 - mask_ratio))

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = th.argsort(noise, dim=1)
        ids_restore = th.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = th.gather(
            x[:, 1:, :], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        x_masked = th.cat([x[:, 0, :].unsqueeze(1), x_masked], dim=1)
        # import pdb 
        # pdb.set_trace()
        # generate the binary mask: 0 is keep, 1 is remove
        mask = th.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = th.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, y, t0, y_lens, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1)
        x_ = th.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        x_ = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x_ = x_ + self.decoder_pos_embed
        x = th.cat([x[:, 0, :].unsqueeze(1), x_], dim=1)
        # import pdb 
        # pdb.set_trace()
        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, y, t0, y_lens, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        # import pdb;pdb.set_trace()
        mask = th.cat([th.ones(mask.shape[0], 1, 1).to(mask.device), mask], dim=1)
        x = x*mask + (1-mask)*x_before
        
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.en_inblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.en_outblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.x_embedder.ol == [0, 0] or self.x_embedder.ol == (0, 0):
            return

        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]

        torch_map = th.zeros(lm+2*lpad, rm+2*rpad).to('cuda')
        for i in range(lf):
            for j in range(rf):
                xx = (i) * (lp - lo)
                yy = (j) * (rp - ro)
                torch_map[xx:(xx+lp), yy:(yy+rp)]+=1 
        torch_map = torch_map[lpad:lm+lpad, rpad:rm+rpad]
        self.torch_map = th.reciprocal(torch_map)

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    

class PixArt_MDT_LC(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=False, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, mask_ratio=None, decode_layer=4,**kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth

        drop_path_half = [x.item() for x in th.linspace(0, drop_path, half_depth)]  # stochastic depth decay rule
        drop_path_decode = [x.item() for x in th.linspace(0, drop_path, decode_layer)]
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp') for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            DEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_decode[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True, FFN_type='mlp', num_f=self.x_embedder.grid_size[1]+1, num_t=self.x_embedder.grid_size[0]) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, FFN_type='mlp') for _ in range(1)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(th.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(th.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.initialize_weights()


    def forward(self, x, timestep, context_list, context_mask_list=None, enable_mask=False, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # import pdb 
        # pdb.set_trace()
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        # if not self.training:
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)
            print("MASK !!!!!!!!!!!!!!!!!!!!!!!!!")

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]

        input_skip = x

        masked_stage = False
        skips = []
        # TODO : masking op for training
        if self.mask_ratio is not None and self.training:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = th.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True


        for block in self.en_inblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and self.training:
            x = self.forward_side_interpolater(x, y, t0, y_lens, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed
        bs = x.shape[0]

        bs, D, L = x.shape
        T, F = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # reshaped = x.reshape(bs, T, F, L).to(x.device)
        end = th.zeros(bs, T, 1, L).to(x.device)
        # x = th.cat((reshaped, zero_tensor), 2)
        # import pdb;pdb.set_trace()
        # assert x.shape == [bs, T, F + 1, L]
        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip
            x, end = auto_grad_checkpoint(block, x, end, y, t0, y_lens, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, context_list, context_mask_list=None, cfg_scale=4.0, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # import pdb
        # pdb.set_trace()
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, context_list, context_mask_list=None)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        if self.x_embedder.ol == (0, 0) or self.x_embedder.ol == [0, 0]:
            c = self.out_channels
            p0 = self.x_embedder.patch_size[0]
            p1 = self.x_embedder.patch_size[1]
            h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

            x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
            x = th.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
            return imgs
    
        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]
        bs = x.shape[0]
        
        torch_map = self.torch_map

        c = self.out_channels

        x = x.reshape(shape=(bs, lf, rf, lp, rp, c))
        x = th.einsum('nhwpqc->nchwpq', x)

        added_map = th.zeros(bs, c, lm+2*lpad, rm+2*rpad).to(x.device)
        
        for i in range(lf):
            for j in range(rf):
                    xx = (i) * (lp - lo)
                    yy = (j) * (rp - ro)
                    added_map[:, :, xx:(xx+lp), yy:(yy+rp)] += \
                    x[:, :, i, j, :, :]
        # import pdb 
        # pdb.set_trace()
        added_map = added_map[:][:][lpad:lm+lpad, rpad:rm+rpad]
        return th.mul(added_map.to(x.device), torch_map.to(x.device))
  
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = th.argsort(noise, dim=1)
        ids_restore = th.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = th.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = th.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = th.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, y, t0, y_lens, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = th.cat([x, mask_tokens], dim=1)
        x = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, y, t0, y_lens, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.en_inblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.en_outblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.x_embedder.ol == [0, 0] or self.x_embedder.ol == (0, 0):
            return

        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]

        torch_map = th.zeros(lm+2*lpad, rm+2*rpad).to('cuda')
        for i in range(lf):
            for j in range(rf):
                xx = (i) * (lp - lo)
                yy = (j) * (rp - ro)
                torch_map[xx:(xx+lp), yy:(yy+rp)]+=1 
        torch_map = torch_map[lpad:lm+lpad, rpad:rm+rpad]
        self.torch_map = th.reciprocal(torch_map)

    @property
    def dtype(self):
        return next(self.parameters()).dtype
   
 
class PixArt_MDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=False, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, mask_ratio=None, decode_layer=4,**kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth

        drop_path_half = [x.item() for x in th.linspace(0, drop_path, half_depth)]  # stochastic depth decay rule
        drop_path_decode = [x.item() for x in th.linspace(0, drop_path, decode_layer)]
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False) for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True) for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_decode[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False) for _ in range(1)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(th.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(th.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.initialize_weights()


    def forward(self, x, timestep, context_list, context_mask_list=None, enable_mask=False, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # print(f'debug_MDT : {x.shape[0]}')
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # import pdb 
        # print(f'debug_MDT : {x.shape[0]}')
        # pdb.set_trace()
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # print(f'debug_MDT : {x.shape[0]}')
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        # print(f'debug_MDT : {x.shape[0]}')
        y = self.y_embedder(y)  # (N, L, D)
        # if not self.training:
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)
            print("MASK !!!!!!!!!!!!!!!!!!!!!!!!!")

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]
        # print(f'debug_MDT : {x.shape[0]}')
        input_skip = x

        masked_stage = False
        skips = []
        # TODO : masking op for training
        if self.mask_ratio is not None and self.training:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = th.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True


        for block in self.en_inblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=None)
    
        if self.mask_ratio is not None and self.training:
            x = self.forward_side_interpolater(x, y, t0, y_lens, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=this_skip, ids_keep=None)
       
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        if self.x_embedder.ol == (0, 0) or self.x_embedder.ol == [0, 0]:
         
            c = self.out_channels
            
            p0 = self.x_embedder.patch_size[0]
            p1 = self.x_embedder.patch_size[1]
            h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

            x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
          
            x = th.einsum('nhwpqc->nchpwq', x)
          
            imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
            
            return imgs
    
        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]
        bs = x.shape[0]
        
        torch_map = self.torch_map

        c = self.out_channels

        x = x.reshape(shape=(bs, lf, rf, lp, rp, c))
        x = th.einsum('nhwpqc->nchwpq', x)

        added_map = th.zeros(bs, c, lm+2*lpad, rm+2*rpad).to(x.device) 
        
        for i in range(lf):
            for j in range(rf):
                    xx = (i) * (lp - lo)
                    yy = (j) * (rp - ro)
                    added_map[:, :, xx:(xx+lp), yy:(yy+rp)] += \
                    x[:, :, i, j, :, :]
                    
        added_map = added_map[:, :, lpad:lm+lpad, rpad:rm+rpad]
        return th.mul(added_map, torch_map.to(added_map.device))

    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = th.argsort(noise, dim=1)
        ids_restore = th.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = th.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = th.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = th.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, y, t0, y_lens, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = th.cat([x, mask_tokens], dim=1)
        x = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, y, t0, y_lens, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.en_inblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.en_outblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.x_embedder.ol == [0, 0] or self.x_embedder.ol == (0, 0):
            return

        lf = self.x_embedder.grid_size[0]
        rf = self.x_embedder.grid_size[1]
        lp = self.x_embedder.patch_size[0]
        rp = self.x_embedder.patch_size[1]
        lo = self.x_embedder.ol[0]
        ro = self.x_embedder.ol[1]
        lm = self.x_embedder.img_size[0]
        rm = self.x_embedder.img_size[1]
        lpad = self.x_embedder.pad_size[0]
        rpad = self.x_embedder.pad_size[1]

        torch_map = th.zeros(lm+2*lpad, rm+2*rpad).to('cuda')
        for i in range(lf):
            for j in range(rf):
                xx = (i) * (lp - lo)
                yy = (j) * (rp - ro)
                torch_map[xx:(xx+lp), yy:(yy+rp)]+=1 
        torch_map = torch_map[lpad:lm+lpad, rpad:rm+rpad]
        self.torch_map = th.reciprocal(torch_map)

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
class PixArt_MDT_FIT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=False, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, mask_ratio=None, decode_layer=4,**kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        decode_layer = int(decode_layer)

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth

        drop_path_half = [x.item() for x in th.linspace(0, drop_path, half_depth)]  # stochastic depth decay rule
        drop_path_decode = [x.item() for x in th.linspace(0, drop_path, decode_layer)]
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False) for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_half[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True) for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_decode[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False, skip=True) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False) for _ in range(1)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(th.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(th.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(th.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.initialize_weights()


    def forward(self, x, timestep, context_list, context_mask_list=None, enable_mask=False, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]
        # import pdb 
        # pdb.set_trace()
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        # if not self.training:
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)
            print("MASK !!!!!!!!!!!!!!!!!!!!!!!!!")

        assert mask is not None
        # if mask is not None:

        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]

        input_skip = x

        masked_stage = False
        skips = []
        # TODO : masking op for training
        if self.mask_ratio is not None and self.training:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = th.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True


        for block in self.en_inblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and self.training:
            x = self.forward_side_interpolater(x, y, t0, y_lens, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = auto_grad_checkpoint(block, x, y, t0, y_lens, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        
        c = self.out_channels
        p0 = self.x_embedder.patch_size[0]
        p1 = self.x_embedder.patch_size[1]
        h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

        x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
        x = th.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = th.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = th.argsort(noise, dim=1)
        ids_restore = th.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = th.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = th.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = th.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, y, t0, y_lens, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = th.cat([x, mask_tokens], dim=1)
        x = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, y, t0, y_lens, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        
        # Replace the absolute embedding with 2d-rope position embedding:
        # pos_embed = 
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.en_inblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.en_outblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class PixArt_Slow(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), patch_size=(16,4), overlap=(0, 0), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0,
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed_1D(input)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size[0] // self.patch_size[0] * 2
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", th.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)
        drop_path = [x.item() for x in th.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock_Slow(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]),
                          window_size=0,
                          use_rel_pos=False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def forward(self, x, timestep, context_list, context_mask_list=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        mask = context_mask_list[0] # (N, L)

        assert mask is not None
        # if mask is not None:

        # y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        # y_lens = mask.sum(dim=1).tolist()
        # y_lens = [int(_) for _ in y_lens]
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, mask)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify(self, x):

        """
        x: (N, T, patch_size 0 * patch_size 1 * C)
        imgs: (Bs. 256. 16. 8)
        """
        c = self.out_channels
        p0 = self.x_embedder.patch_size[0]
        p1 = self.x_embedder.patch_size[1]
        h, w = self.x_embedder.grid_size[0], self.x_embedder.grid_size[1]

        x = x.reshape(shape=(x.shape[0], h, w, p0, p1, c))
        x = th.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p0, w * p1))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.cross_attn.proj.weight, 0)
        #     nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class PixArtBlock_1D(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=None,
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(th.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        # x [3, 133, 1152]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

class PixArt_1D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0, 
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed_1D(input_size, in_channels, hidden_size)
        # self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.p_enc_1d_model = PositionalEncoding1D(hidden_size)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)
        drop_path = [x.item() for x in th.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock_1D(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          window_size=0,
                          use_rel_pos=False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, (1, input_size[1]), self.out_channels)

        self.initialize_weights()

        # if config:
        #     logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        #     logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        # else:
        #     print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, context_list, context_mask_list=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)

        x = self.x_embedder(x)  # (N, T, D)
        pos_embed = self.p_enc_1d_model(x)
        x = x + pos_embed
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        try:
            mask = context_mask_list[0] # (N, L)
        except:
            mask = th.ones(x.shape[0], 1).to(x.device)
            print("MASK !!!!!!!!!!!!!!!!!!!!!!!!!")

        assert mask is not None
        # if mask is not None:
        y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        y_lens = mask.sum(dim=1).tolist()
        y_lens = [int(_) for _ in y_lens]
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify_1D(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify_1D(self, x):

        """
        """
        c = self.out_channels

        x = x.reshape(shape=(x.shape[0], self.input_size[0], self.input_size[1], c))
        x = th.einsum('btfc->bctf', x)
        # imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        # self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
class PixArt_Slow_1D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=(256,16), in_channels=8, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, cond_dim=1024, lewei_scale=1.0, 
                 use_cfg=True, cfg_scale=4.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        self.input_size = input_size
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed_1D(input_size, in_channels, hidden_size)
        # self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.p_enc_1d_model = PositionalEncoding1D(hidden_size)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = nn.Linear(cond_dim, hidden_size)
        drop_path = [x.item() for x in th.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock_Slow(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          window_size=0,
                          use_rel_pos=False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, (1, input_size[1]), self.out_channels)

        self.initialize_weights()

        # if config:
        #     logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        #     logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        # else:
        #     print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, context_list, context_mask_list=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = context_list[0].to(self.dtype)

        x = self.x_embedder(x)  # (N, T, D)
        pos_embed = self.p_enc_1d_model(x)
        x = x + pos_embed
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)
        mask = context_mask_list[0] # (N, L)

        assert mask is not None
        # if mask is not None:
        # y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        # y_lens = mask.sum(dim=1).tolist()
        # y_lens = [int(_) for _ in y_lens]
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, mask)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify_1D(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps
        # return th.cat([eps, rest], dim=1)

    def unpatchify_1D(self, x):

        """
        """
        c = self.out_channels

        x = x.reshape(shape=(x.shape[0], self.input_size[0], self.input_size[1], c))
        x = th.einsum('btfc->bctf', x)
        # imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, lewei_scale=self.lewei_scale, base_size=self.base_size)
        # self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.cross_attn.proj.weight, 0)
        #     nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # import pdb
    # pdb.set_trace()
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)

# if __name__ == '__main__' :
#     import pdb
#     pdb.set_trace()
#     model = PixArt_1D().to('cuda')
#     # x: (N, T, patch_size 0 * patch_size 1 * C)
#     th.manual_seed(233)
#     # x = th.rand(1, 4*16, 16*4*16).to('cuda')
#     x = th.rand(3, 8, 256, 16).to('cuda')
    
#     t = th.tensor([1, 2, 3]).to('cuda')
#     c = th.rand(3, 20, 1024).to('cuda')
#     c_mask = th.ones(3, 20).to('cuda')
#     c_list = [c]
#     c_mask_list = [c_mask]
#     y = model.forward(x, t, c_list, c_mask_list)
    # res = model.unpatchify(x)
# class DiTModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     :param num_heads_channels: if specified, ignore num_heads and instead use
#                                a fixed channel width per attention head.
#     :param num_heads_upsample: works with num_heads to set a different number
#                                of heads for upsampling. Deprecated.
#     :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
#     :param resblock_updown: use residual blocks for up/downsampling.
#     :param use_new_attention_order: use a different attention pattern for potentially
#                                     increased efficiency.
#     """

#     def __init__(
#         self,
#         input_size,
#         patch_size,
#         overlap,
#         in_channels,
#         embed_dim,
#         model_channels,
#         out_channels,
#         dims=2,
#         extra_film_condition_dim=None,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=-1,
#         num_head_channels=-1,
#         use_scale_shift_norm=False,
#         use_new_attention_order=False,
#         transformer_depth=1,  # custom transformer support
#         context_dim=None,  # custom transformer support
#         n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
#         legacy=True,
#     ):
#         super().__init__()
#         self.x_embedder = PatchEmbed(input_size, patch_size, overlap, in_channels, embed_dim, bias=True)
#         num_patches = self.x_embedder.num_patches
#         self.pos_embed = nn.Parameter(th.zeros(1, num_patches, embed_dim), requires_grad=False)
#         self.blocks = nn.ModuleList([
#             DiTBlock_crossattn
#         ])

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         # self.input_blocks.apply(convert_module_to_f16)
#         # self.middle_block.apply(convert_module_to_f16)
#         # self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         # self.input_blocks.apply(convert_module_to_f32)
#         # self.middle_block.apply(convert_module_to_f32)
#         # self.output_blocks.apply(convert_module_to_f32)

#     def forward(
#         self,
#         x,
#         timesteps=None,
#         y=None,
#         context_list=None,
#         context_attn_mask_list=None,
#         **kwargs,
#     ):
#         """
#         Apply the model to an input batch.
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param context: conditioning plugged in via crossattn
#         :param y: an [N] Tensor of labels, if class-conditional. an [N, extra_film_condition_dim] Tensor if film-embed conditional
#         :return: an [N x C x ...] Tensor of outputs.
#         """
        
#         x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
#         t = self.t_embedder(timesteps)                   # (N, D)
#         y = self.y_embedder(y, self.training)    # (N, D)
#         c = t + y                                # (N, D)
#         for block in self.blocks:
#             x = block(x, c)                      # (N, T, D)
#         x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
#         x = self.unpatchify(x)                   # (N, out_channels, H, W)
        

