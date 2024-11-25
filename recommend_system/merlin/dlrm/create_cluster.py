#!/usr/bin/python3

import numba
import warnings
from dask_cuda import LocalCUDACluster
import nvtabular as nvt
from merlin.core.compat import pynvml_mem_size, device_mem_size

def load_cluster(device_limit_frac = 0.7, device_pool_frac = 0.8, dask_workdir = 'workdir', dashboard_port = 8787):
  assert numba.cuda.is_available()
  cluster = LocalCUDACluster(
    protocal = 'tcp',
    n_workers = len(numba.cuda.gpus),
    CUDA_VISIBLE_DEVICES=','.join([str(gpu) for gpu in range(len(numba.cuda.gpus))]),
    device_memory_limit = int(device_limit_frac * len(numba.cuda.gpus)),
    local_directory = dask_workdir,
    dashboard_address = f":{dashboard_port}",
    rmm_pool_size = (int(device_pool_frac * len(numba.cuda.gpus)) // 256) * 256
  )
  return cluster
