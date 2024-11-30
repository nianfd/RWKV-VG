import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from .vl_rwkv import get_model_RWKV_CLIP
from utils.box_utils import xywh2xyxy
from model.utils import *
from model import VL_RWKV

# class TransVG(nn.Module):
#     def __init__(self, args, argsRWKV):
#         super(TransVG, self).__init__()
        
#         hidden_dim = args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.num_visu_token = int((args.imsize / divisor) ** 2)
#         #print(self.num_visu_token) #400
#         self.num_text_token = args.max_query_len
#         #print('TransVG1')
#         #self.visumodel = build_detr(args)
#         #print('TransVG')
#         #self.textmodel = build_bert(args)
        
#         self.vl_rwkvmodel = get_model_RWKV_CLIP(argsRWKV)
#         #self.vl_rwkvmodel.eval()
#         self.vl_rwkvmodel.cuda()
        
#         #print('vl_rwkvmodel build finish')
#         num_total = self.num_visu_token + self.num_text_token + 1
#         self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
#         self.reg_token = nn.Embedding(1, hidden_dim)

#         self.visu_proj = nn.Linear(640, hidden_dim)#self.visumodel.num_channels:256
#         #self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
#         self.text_proj = nn.Linear(640, hidden_dim)
        
#         #self.vl_transformer = build_vl_transformer(args)
#         self.vl_rwkvREG = VL_RWKV()
#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

#     def forward(self, img_data, text_data):
#         #print('heihei')
#         bs = img_data.tensors.shape[0]

#         # visual backbone
#         #visu_mask, visu_src = self.visumodel(img_data)
#         # print(visu_mask.shape) #40*400, 40 is batchsize
#         # print(visu_src.shape) #400*40*256  40 is batchsize
#         #print(self.visumodel.num_channels) 256
#         visu_src = WarperCLIP_V_T_RWKV_method(self.vl_rwkvmodel, img_data.tensors)[0]
#         visu_src = visu_src.view(bs, 400, 640)
#         visu_src = visu_src.permute(1, 0, 2)
#         #print(visu_src.shape) #torch.Size([40, 20, 20, 640])
#         visu_src = self.visu_proj(visu_src) # (N*B)xC
#         #print(visu_src.shape) torch.Size([400, 8, 256]), 8 is batchsize, 400 is visual token, 256 is feature dim
#         # language bert
#         # print(text_data.tensors)
#         # print(text_data.mask)
#         #text_fea = self.textmodel(text_data)
#         #print(text_fea)
#         # visu_src_nfd = WarperCLIP_V_T_RWKV_method(self.vl_rwkvmodel, img_data.tensors)
#         # print(visu_src_nfd.shape)
        
#         text_src = WarperCLIP_V_T_RWKV_text(self.vl_rwkvmodel, text_data.tensors)
#         mask = text_data.mask.to(torch.bool)
#         text_mask = ~mask
#         #text_src, text_mask = text_fea.decompose()
#         # print(text_src.shape) #torch.Size([8, 20, 768])
#         # print(text_mask.shape)#torch.Size([8, 20])
#         # print(text_mask)
#         #print('=======')
#         assert text_mask is not None
#         text_src = self.text_proj(text_src)
#         #print(text_src.shape) #torch.Size([8, 20, 256])
#         # permute BxLenxC to LenxBxC
#         text_src = text_src.permute(1, 0, 2)
#         #print(text_src.shape) #torch.Size([20, 40, 256])
#         text_mask = text_mask.flatten(1)
#         #print(text_mask)

#         # target regression token
#         tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
#         tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
#         #print(tgt_src.shape) #torch.Size([1, 40, 256]) #40 is batch size

#         vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
#         #vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
#         vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
#         #print(vl_pos.shape) #torch.Size([421, 24, 256]) #24 is batch size, includes [REG] token

#         vg_hs = self.vl_transformer(vl_src, None, vl_pos) # (1+L+N)xBxC

#         #print(vg_hs.shape)
#         #vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
#         vg_hs = vg_hs[0]
#         #print(vg_hs.shape)  torch.Size([24, 256])  #get first token representation

#         pred_box = self.bbox_embed(vg_hs).sigmoid()

#         return pred_box

class TransVG(nn.Module):
    def __init__(self, args, argsRWKV):
        super(TransVG, self).__init__()
        
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        #print(self.num_visu_token) #400
        self.num_text_token = args.max_query_len
        #print('TransVG1')
        #self.visumodel = build_detr(args)
        #print('TransVG')
        #self.textmodel = build_bert(args)
        
        self.vl_rwkvmodel = get_model_RWKV_CLIP(argsRWKV)
        #self.vl_rwkvmodel.eval()
        self.vl_rwkvmodel.cuda()
        
        #print('vl_rwkvmodel build finish')
        #num_total = self.num_visu_token + self.num_text_token + 1
        #self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        #self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(640, hidden_dim)#self.visumodel.num_channels:256
        #self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(640, hidden_dim)
        
        #self.vl_transformer = build_vl_transformer(args)
        self.vl_rwkvREG = VL_RWKV(argsRWKV)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

    def forward(self, img_data, text_data):
        #print('heihei')
        bs = img_data.tensors.shape[0]

        # visual backbone
        #visu_mask, visu_src = self.visumodel(img_data)
        # print(visu_mask.shape) #40*400, 40 is batchsize
        # print(visu_src.shape) #400*40*256  40 is batchsize
        #print(self.visumodel.num_channels) 256
        visu_src = WarperCLIP_V_T_RWKV_method(self.vl_rwkvmodel, img_data.tensors)[0]
        visu_src = visu_src.view(bs, 400, 640)
        visu_src = visu_src.permute(1, 0, 2)
        #print(visu_src.shape) #torch.Size([40, 20, 20, 640])
        visu_src = self.visu_proj(visu_src) # (N*B)xC
        #print(visu_src.shape) torch.Size([400, 8, 256]), 8 is batchsize, 400 is visual token, 256 is feature dim
        # language bert
        # print(text_data.tensors)
        # print(text_data.mask)
        #text_fea = self.textmodel(text_data)
        #print(text_fea)
        # visu_src_nfd = WarperCLIP_V_T_RWKV_method(self.vl_rwkvmodel, img_data.tensors)
        # print(visu_src_nfd.shape)
        
        text_src = WarperCLIP_V_T_RWKV_text(self.vl_rwkvmodel, text_data.tensors)
        mask = text_data.mask.to(torch.bool)
        text_mask = ~mask
        #text_src, text_mask = text_fea.decompose()
        # print(text_src.shape) #torch.Size([8, 20, 768])
        # print(text_mask.shape)#torch.Size([8, 20])
        # print(text_mask)
        #print('=======')
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        #print(text_src.shape) #torch.Size([8, 20, 256])
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        #print(text_src.shape) #torch.Size([20, 40, 256])#40 is batch size
        text_mask = text_mask.flatten(1)
        #print(text_mask)

        # target regression token
        #tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        #tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        #print(tgt_src.shape) #torch.Size([1, 40, 256]) #40 is batch size

        vl_src = torch.cat([text_src, visu_src], dim=0)
        #vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        #vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        #print(vl_pos.shape) #torch.Size([421, 24, 256]) #24 is batch size, includes [REG] token

        #vg_hs = self.vl_transformer(vl_src, None, vl_pos) # (1+L+N)xBxC
        vg_hs = self.vl_rwkvREG(vl_src) # (1+L+N)xBxC
        #print(vg_hs.shape)
        #vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[-1]  #REG token end
        
        #vg_hs = vg_hs[0]   #REG token start
        #print(vg_hs.shape)  #torch.Size([24, 256])  #get first token representation

        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
