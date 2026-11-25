import os

import torch

# Without a GPU, run kernels in Triton's interpreter (must be set before triton is imported).
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
