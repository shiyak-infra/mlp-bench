import requests

BENCHMARK_GPU_COMPUTATION_PRECISION = "gpu_computation_precision"
BENCHMARK_GPU_FLOPS = "gpu_flops"
BENCHMARK_NVLINK = "nvlink"
BENCHMARK_RDMA = "rdma"

available_benchmark_names = {BENCHMARK_GPU_COMPUTATION_PRECISION, BENCHMARK_GPU_FLOPS, BENCHMARK_NVLINK, BENCHMARK_RDMA}


def get_config(driver_master_host: str):
    try:
        response = requests.get(f"{driver_master_host}/api/v1/config")
        if response.status_code != 200:
            raise Exception(f"invalid status code {response.status_code}, body: {str(response.text)}")
        return response.json()
    except Exception as e:
        raise e
