import time
import torch
import logging
import numpy as np

from modeling.openclip_model import OpenCLIPModel as OPT_Model
from clip_server.model.openclip_model import OpenCLIPModel as ORG_Model


def benchmark(N = 1, B = 1):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model: mock input
    name='ViT-L-14::laion2b-s32b-b82k'
    opt_model = OPT_Model(name=name, device='cuda')
    org_model = ORG_Model(name=name, device='cuda')

    # Benchmark
    complete_time_baseline = 0
    complete_time_optimized = 0
    
    # input = torch.randint(0, 10, (B, 77)).long().cuda()
    input = torch.randint(0, 10, (B, 3, 224, 224)).half().cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        # warm up
        for _ in range(10):
            _1 = org_model.encode_image(input)
            _2 = opt_model.encode_image(input)
        
        # benchamrk
        for _ in range(N):

            torch.cuda.synchronize()
            start = time.perf_counter()
            _1 = org_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_baseline += time.perf_counter() - start

            torch.cuda.synchronize()
            start = time.perf_counter()
            _2 = opt_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start

    print(f"{complete_time_baseline=:.5f}s")
    print(f"{complete_time_optimized=:.5f}s")
    mean_diff = np.mean(abs(_1.cpu().numpy()-_2.cpu().numpy()))
    print(f"{mean_diff}")
    return complete_time_baseline, complete_time_optimized, mean_diff
    

if __name__ == "__main__":
    import time
    complete_time_baseline = []
    complete_time_optimized = []
    mean_diff = []
    speed_up = []
    for N in [100]:
        for B in [1, 2, 4, 8, 16]:
            print(f"Runing on N={N}, B={B}")
            complete_time_baseline_, complete_time_optimized_, mean_diff_ = benchmark(N, B)
            complete_time_baseline.append(complete_time_baseline_)
            complete_time_optimized.append(complete_time_optimized_)
            mean_diff.append(mean_diff_)
            speed_up_ = complete_time_baseline_/complete_time_optimized_
            speed_up.append(speed_up_)
            print(f"Speed up:{speed_up_}")
            print(f"Diff:{mean_diff_}\n")
    print(f"{complete_time_baseline}\n")
    print(f"{complete_time_optimized}\n")
    print(f"{mean_diff}\n")
    print(f"{speed_up}\n")