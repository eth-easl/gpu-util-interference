# Pytorch Matrix Multiplication - Interference with ML kernel
This script contains the supporting code for section 4.3 of our [report](https://arxiv.org/pdf/2501.16909). The benchmark measures interference between PyTorch's [matrix multiplication](https://pytorch.org/docs/stable/generated/torch.mm.html) kernel and a custom compute kernel `fma_fp32_ilp4`.

### Running the paper experiment
Before running the experiment, create a virtual environment and install pytorch
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

The main experiment can be run with
```bash
python3 run_mm_pytorch_interf.py
```
The script compares
- individual latencies when running the PyTorch mm and fp32_fma_ilp4 kernel alone
- latencies when running the PyTorch mm and fp32_fma_ilp4 kernel colocated using CUDA streams

We are using [CTypes](https://docs.python.org/3/library/ctypes.html) to call our custom `fma_fp32_ilp4` kernel from Python. We use the custom kernel in order to generate interference at the IPC and FMA pipeline level.

To visualize kernel overlap and latency impact it is very useful to collect the CUDA trace with the Nsight Systems profiler.
```bash
nsys profile --force-overwrite true -o mm_pytorch_interf.nsys-rep --trace cuda python3 run_mm_pytorch_interf.py
```

### Relevant NCU metrics
For this experiment the relevant NCU metrics are:
1. Pipe utilization (% of peak instructions executed)
    - `sm__inst_executed_<pipeline>_fma.avg.pct_of_peak_sustained_active`
2. Average issued instructions per cycle per SM while active [instr/cycle]
    - `sm__inst_issued.avg.per_cycle_active`
3. Average issue slots busy [%]
    - `sm__inst_issued.avg.pct_of_peak_sustained_active`