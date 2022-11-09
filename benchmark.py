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
    
    input = torch.randint(0, 10, (B, 77)).long().cuda()
    # inputs_image = torch.randint(0, 10, (N, B, 3, 224, 224)).half().cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for _ in range(N):

            torch.cuda.synchronize()
            start = time.perf_counter()
            _1 = org_model.encode_text(input)
            torch.cuda.synchronize()
            complete_time_baseline += time.perf_counter() - start

            torch.cuda.synchronize()
            start = time.perf_counter()
            _2 = opt_model.encode_text(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start

    print(f"{complete_time_baseline=:.5f}s")
    print(f"{complete_time_optimized=:.5f}s")
    show_diff(_1, _2)
    return complete_time_baseline, complete_time_optimized
    

def show_diff(a, b):
    from matplotlib import pyplot as plt
    # print(a)
    # print(b)
    
    a = a.cpu().numpy()[0]
    b = b.cpu().numpy()[0]
    plt.plot(np.arange(768), a-b)
    plt.show()
    

if __name__ == "__main__":
    import time
    complete_time_baseline = []
    complete_time_optimized = []
    speed_up = []
    # warm up
    for _ in range(10):
        _, _ = benchmark(1, 1)
    # benchmark
    for N in [1, 100, 1000, 5000, 10000]:
        for B in [1, 2, 4, 8, 16]:
            print(f"Runing on N={N}, B={B}")
            complete_time_baseline_, complete_time_optimized_ = benchmark(N, B)
            complete_time_baseline.append(complete_time_baseline_)
            complete_time_optimized.append(complete_time_optimized_)
            speed_up_ = complete_time_baseline_/complete_time_optimized_
            speed_up.append(speed_up_)
            print(f"Speed up:{speed_up_}\n")
    print(complete_time_baseline)
    print(complete_time_optimized)
    print(speed_up)

    np.savetxt('./assets/baseline.txt', complete_time_baseline)
    np.savetxt('./assets/optimized.txt', complete_time_optimized)
    np.savetxt('./assets/speed_up.txt', speed_up)
