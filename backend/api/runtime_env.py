"""
Configure native ML/OpenCV threading before heavy libraries load.
Reduces Windows access-violation crashes (exit 0xC0000005) when PyTorch,
OpenCV, and OpenMP run in the same process as Uvicorn.
"""
import os

_NATIVE_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "TOKENIZERS_PARALLELISM": "false",
}

for _key, _val in _NATIVE_ENV.items():
    os.environ.setdefault(_key, _val)
