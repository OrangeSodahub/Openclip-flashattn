import torch
from torch.profiler import profile, record_function, ProfilerActivity

from modeling.openclip_model import OpenCLIPModel as OPT_Model
from clip_server.model.openclip_model import OpenCLIPModel as ORG_Model


def profiler(name, mode, B):
    # Load Model: mock input
    opt_model = OPT_Model(name=name, device='cuda')
    org_model = ORG_Model(name=name, device='cuda')

    # setup inputs
    if mode == 'text':
        input = torch.randint(0, 10, (B, 77)).long().cuda()
    elif mode == 'image':
        input = torch.randint(0, 10, (B, 3, 224, 224)).half().cuda()

    # setup encode fn
    if mode == 'text':
        org_encode = org_model.encode_text
        opt_encode = opt_model.encode_text
    elif mode == 'image':
        org_encode = org_model.encode_image
        opt_encode = opt_model.encode_image
    
    # warm up
    for _ in range(10):
        _1 = org_encode(input)
        _2 = opt_encode(input)

    # profile time cosumption
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            _1 = org_encode(input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):    
            _2 = opt_encode(input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    name = 'ViT-B-16::laion400m_e31'
    
    profiler(name, 'text', 8)