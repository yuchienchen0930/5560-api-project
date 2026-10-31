import os
# Anaconda and PyTorch each ship their own OpenMP runtime; avoid the OMP Error #15 crash on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
