"""Different tasks implemented."""

import os
import warnings
from torch.jit._trace import TracerWarning

# Removing harmless warnings from the tracer and wrong bottleneck identification
warnings.simplefilter("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", message=".*bottleneck.*num_workers.*")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
