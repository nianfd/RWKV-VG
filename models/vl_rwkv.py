from model import Text_RWKV, Image_RWKV, get_model, VL_RWKV
import torch
import torch.nn.functional as F
import torch.nn as nn
def load_model_weight(model, model_weight):
    state_dict = torch.load(model_weight)
    #state_dict.pop('module.image_model.pos_embed')
    #print(state_dict.items())
    state_dict_removed = {}
    for k, value in state_dict.items():
        #print(k)
        k_removed = k
        if "module." in k_removed:
            k_removed = k.split("module.")[-1]
        if '_orig_mod.' in k_removed:
            k_removed = k_removed.split('_orig_mod.')[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k_removed] = value
    #print(state_dict_removed)
    model.load_state_dict(state_dict_removed, strict=False)
    #print(model.image_model.pos_embed.shape)
    # 增加一个通道维度，使其变为 [1, 1, 49, 640]
    input_tensor = model.image_model.pos_embed.unsqueeze(1)

    # 使用 F.interpolate 进行插值
    output_tensor = F.interpolate(input_tensor, size=(400, 640), mode='bilinear', align_corners=False)

    model.image_model.pos_embedLarge = nn.Parameter(output_tensor.squeeze(1))
    return model  

def get_model_RWKV_CLIP(args):
    model_image_rwkv = Image_RWKV(img_size = args.input_size,
                            patch_size= args.image_patch_size,
                            embed_dims = args.image_embed_dims, 
                            hidden_rate= args.image_hidden_rate, 
                            depth=args.image_depth,
                            num_heads=args.image_num_heads,
                            output_cls_token=args.image_output_cls_token,
                            with_cls_token=args.image_with_cls_token)
    #print('model_image_rwkv finish')        
    model_text_rwkv = Text_RWKV(args)
    model = get_model(model_image_rwkv, model_text_rwkv, image_cls_token=args.image_output_cls_token)
    if args.model_weight !='':
        print('load pretrained rwkv-clip')
        model = load_model_weight(model, args.model_weight)
    else:
        print(' not load pretrained rwkv-clip')
    return model