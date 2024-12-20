# Copyright (c) Shanghai AI Lab OpenGV lab. All rights reserved.
import os
import torch
import torch.nn as nn
from typing import Sequence
import torch.utils.checkpoint as cp
from torch.nn import functional as F

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone
from model.utils_vision_rwkv.drop import DropPath

T_MAX = int(float(os.environ['Image_T_max']))
HEAD_SIZE = int(os.environ['Image_HEAD_SIE'])

from torch.utils.cpp_extension import load

########################################################################################################
# CUDA Kernel
########################################################################################################
wkv6_cuda = load(name="wkv6_image",
                sources=["model/cuda_image/wkv6_op.cpp", 
                        "model/cuda_image/wkv6_cuda.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                "-O3", "-Xptxas -O3", 
                "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", 
                f"-D_T_={T_MAX}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            
            wkv6_cuda.forward(B, T, C, H, r.float(), k.float(), v.float(), ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            
            wkv6_cuda.backward(B, T, C, H, r.float(), k.float(), v.float(), ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################
# q shift
########################################################################################################
def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE, 
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim*1/4), :, shift_pixel:W] = input[:, :, 0:int(head_dim*1/4), :, 0:W-shift_pixel]
    output[:, :, int(head_dim/4):int(head_dim/2), :, 0:W-shift_pixel] = input[:, :, int(head_dim/4):int(head_dim/2), :, shift_pixel:W]
    output[:, :, int(head_dim/2):int(head_dim/4*3), shift_pixel:H, :] = input[:, :, int(head_dim/2):int(head_dim/4*3), 0:H-shift_pixel, :]
    output[:, :, int(head_dim*3/4):int(head_dim), 0:H-shift_pixel, :] = input[:, :, int(head_dim*3/4):int(head_dim), shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N-1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output


########################################################################################################
# VRWKV_SpatialMix
########################################################################################################
class VRWKV_SpatialMix_V6(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False, 
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        
        #print(self.attn_sz)
        
        # print(self.head_size)
        # print(HEAD_SIZE)
        
        
        #assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.shift_func = eval(shift_mode)

        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    ddd[0, 0, i] = i / self.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.attn_sz))

                TIME_DECAY_EXTRA_DIM = 64
                self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()

        xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution, 
                             with_cls_token=self.with_cls_token) - x # direct follow Vision-RWKV
        xxx = x + xx * self.time_maa_x  # [B, T, C]
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        # [5, B*T, TIME_MIX_EXTRA_DIM]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # [5, B, T, C]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            r, k, v, g, w = self.jit_func(x, patch_resolution)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            if self.key_norm is not None:
                x = self.key_norm(x)
            return self.jit_func_2(x, g)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


########################################################################################################
# VRWKV_ChannelMix
########################################################################################################
class VRWKV_ChannelMix_V6(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, 
                 with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.shift_func = eval(shift_mode)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                                 with_cls_token=self.with_cls_token)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


########################################################################################################
# Blocks
########################################################################################################
class Block_V6(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                       shift_pixel, init_mode, key_norm=key_norm,
                                       with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cls_token=with_cls_token)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Image_RWKV(BaseBackbone):
    def __init__(self,
                 img_size=640,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=192,
                 num_heads=3,
                 depth=12,
                 drop_path_rate=0.,
                 shift_pixel=1,
                 shift_mode='q_shift_multihead',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 output_cls_token=False,
                 with_cls_token=False,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        #print(num_patches)
        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 49, self.embed_dims))
        #print(self.pos_embed.shape) #torch.Size([1, 400, 640])
        self.pos_embedLarge = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block_V6(
                n_embd=embed_dims,
                n_head=num_heads,
                n_layer=depth,
                layer_id=i,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cls_token=with_cls_token,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        #print(patch_resolution)
        #print(x.shape)
        
        #print(self.with_cls_token)
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)  # post cls_token

        x = self.drop_after_pos(x)
        #print(x.shape)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)
            #print(i)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)
            #print(self.out_indices)
            
            if i in self.out_indices:
                B, _, C = x.shape
                #print(x.shape)
                if self.with_cls_token:
                    patch_token = x[:, :-1].reshape(B, *patch_resolution, C)
                    
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, -1]
                else:
                    #print(x.shape) #torch.Size([24, 400, 640])
                    patch_token = x.reshape(B, *patch_resolution, C)
                    #print(patch_token.shape) #torch.Size([24, 20, 20, 640])
                    #patch_token = patch_token.permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                #print(out.shape)
                outs.append(out)
        #print(outs[0].shape)
        return tuple(outs)
    
    
# class VL_RWKV(BaseBackbone):
#     def __init__(self,
#                  out_indices=-1,
#                  drop_rate=0.,
#                  embed_dims=640,
#                  num_heads=3,
#                  depth=12,
#                  drop_path_rate=0.,
#                  shift_pixel=1,
#                  shift_mode='q_shift_multihead',
#                  init_mode='fancy',
#                  post_norm=False,
#                  key_norm=False,
#                  init_values=None,
#                  hidden_rate=4,
#                  final_norm=True,
#                  output_cls_token=True,
#                  with_cls_token=True,
#                  with_cp=False,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         self.embed_dims = embed_dims
#         self.num_extra_tokens = 0
#         self.num_layers = depth
#         self.drop_path_rate = drop_path_rate

#         # Set cls token
#         if output_cls_token:
#             assert with_cls_token is True, f'with_cls_token must be True if' \
#                 f'set output_cls_token to True, but got {with_cls_token}'
#         self.with_cls_token = with_cls_token
#         self.output_cls_token = output_cls_token
#         if self.with_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
#         self.pos_embedLarge = nn.Parameter(
#             torch.zeros(1, 400, self.embed_dims))
#         self.drop_after_pos = nn.Dropout(p=drop_rate)

#         if isinstance(out_indices, int):
#             out_indices = [out_indices]
#         assert isinstance(out_indices, Sequence), \
#             f'"out_indices" must by a sequence or int, ' \
#             f'get {type(out_indices)} instead.'
#         for i, index in enumerate(out_indices):
#             if index < 0:
#                 out_indices[i] = self.num_layers + index
#             assert 0 <= out_indices[i] <= self.num_layers, \
#                 f'Invalid out_indices {index}'
#         self.out_indices = out_indices
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.layers = ModuleList()
#         for i in range(self.num_layers):
#             print(str(i) +  ': layer')
#             self.layers.append(Block_V6(
#                 n_embd=embed_dims,
#                 n_head=num_heads,
#                 n_layer=depth,
#                 layer_id=i,
#                 shift_pixel=shift_pixel,
#                 shift_mode=shift_mode,
#                 hidden_rate=hidden_rate,
#                 drop_path=dpr[i],
#                 init_mode=init_mode,
#                 post_norm=post_norm,
#                 key_norm=key_norm,
#                 init_values=init_values,
#                 with_cls_token=with_cls_token,
#                 with_cp=with_cp
#             ))

#         self.final_norm = final_norm
#         if final_norm:
#             self.ln1 = nn.LayerNorm(self.embed_dims)

#     def forward(self, x):
#         B = x.shape[0]
#         #print(patch_resolution)
#         #print(x.shape)
#         x = x + self.pos_embedLarge
#         #print(self.with_cls_token)
#         if self.with_cls_token:
#             cls_tokens = self.cls_token.expand(B, -1, -1)
#             x = torch.cat((x, cls_tokens), dim=1)  # post cls_token

#         x = self.drop_after_pos(x)
#         #print(x.shape)
#         outs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             #print(i)
#             if i == len(self.layers) - 1 and self.final_norm:
#                 x = self.ln1(x)
#             #print(self.out_indices)
            
#             if i in self.out_indices:
#                 B, _, C = x.shape
#                 #print(x.shape)
#                 if self.with_cls_token:
#                     # patch_token = x[:, :-1].reshape(B, *patch_resolution, C)
#                     # patch_token = patch_token.permute(0, 3, 1, 2)
#                     cls_token = x[:, -1]
#                 else:
#                     patch_token = x.reshape(B, *patch_resolution, C)
#                     #patch_token = patch_token.permute(0, 3, 1, 2)
#                 if self.output_cls_token:
#                     out = cls_token
#                 else:
#                     out = patch_token
#                 #print(out.shape)
#                 outs.append(out)
#         #print(outs[0].shape)
#         return tuple(outs)