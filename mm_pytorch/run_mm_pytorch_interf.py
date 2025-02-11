import argparse
import torch
import threading

from ctypes import *


def alloc_mats(dim1, dim2, dim3):
    mat1 = torch.randn(dim1, dim2, pin_memory=True).cuda(non_blocking=True)
    mat2 = torch.randn(dim2, dim3, pin_memory=True).cuda(non_blocking=True)
    return mat1, mat2


def run_mm(m11, m12, num_runs, stream):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.cuda.stream(stream):
        for i in range(num_runs):
            start_events[i].record()
            torch.mm(m11, m12)
            end_events[i].record()

    # need to synchronize at the very end to make sure, all operations end up in nsys trace
    end_events[-1].synchronize()

    lats = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
    sort_lats = sorted(lats)

    if num_runs > 1:
        mid = num_runs // 2
        avg_lat = (sort_lats[mid] + sort_lats[mid + 1]) / 2
    else:
        avg_lat = sort_lats[0]

    for i in range(num_runs):
        print(f"[MM PyTorch] Run {i}, Latency: {lats[i]} ms")
    print(f"AVG MED time is {avg_lat} ms")

    return avg_lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024, help="Matrix A rows")
    parser.add_argument(
        "--k", type=int, default=1024, help="Matrix A cols / Matrix B rows"
    )
    parser.add_argument("--n", type=int, default=1024, help="Matrix B cols")
    parser.add_argument(
        "--iters_interf",
        type=int,
        default=300000,
        help="Number of iterations for the interference kernel",
    )
    parser.add_argument(
        "--runs_mm", type=int, default=100, help="Number of runs for the mm kernel"
    )
    parser.add_argument(
        "--runs_interf",
        type=int,
        default=4,
        help="Number of runs for the interference kernel",
    )
    parser.add_argument(
        "--shared_lib",
        type=str,
        default="./../build/mm_pytorch/libpython_interface.so",
        help="Path to the shared library",
    )
    args = parser.parse_args()

    c_funcs = CDLL(args.shared_lib)
    print("Python C interface shared library loaded")

    stream = torch.cuda.Stream()

    # run the mm kernel once, to preload it into memory
    # otherwise risk of lazy loading leading to sequential kernel execution (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#possible-issues-when-adopting-lazy-loading)
    m11, m12 = alloc_mats(args.m, args.k, args.n)
    run_mm(m11, m12, 1, stream)
    torch.cuda.synchronize()

    num_tb = 132
    num_threads = 128
    launch_config = f"fp32_fma launch Config: ({num_tb}, {num_threads})"

    print("------------------")
    print("Running MM PyTorch alone")
    run_mm(m11, m12, args.runs_mm, stream)

    print("------------------")
    print(f"Running fp32_fma kernel alone - {launch_config}")
    c_funcs.run_fp32_fma_kernel(
        num_tb, num_threads, args.iters_interf, args.runs_interf
    )

    print("------------------")
    print(f"Running MM PyTorch and fp32_fma kernel collocated - {launch_config}")
    interf_thread = threading.Thread(
        target=c_funcs.run_fp32_fma_kernel,
        args=(num_tb, num_threads, args.iters_interf, args.runs_interf),
    )
    mm_thread = threading.Thread(target=run_mm, args=(m11, m12, args.runs_mm, stream))

    interf_thread.start()
    mm_thread.start()

    mm_thread.join()
    interf_thread.join()

    print("Done!")
