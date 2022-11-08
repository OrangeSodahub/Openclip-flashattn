import torch
from copy import deepcopy

from modeling.model import CLIP
from clip_server.model.model import _MODEL_CONFIGS, load_state_dict, convert_weights_to_fp16


def load_openclip_model(
    model_name: str,
    model_path: str,
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model_name = model_name.replace(
        '/', '-'
    )  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert (
                False
            ), 'pretrained image towers currently only supported for timm models'

    model = CLIP(**model_cfg)
    model.eval()

    model.load_state_dict(load_state_dict(model_path))

    if str(device).startswith('cuda'):
        convert_weights_to_fp16(model)

    model.to(device=device)

    if jit:
        model = torch.jit.script(model)

    return model