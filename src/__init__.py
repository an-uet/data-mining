"""Beijing PM2.5 (Aotizhongxin) HW1: next-hour PM2.5 prediction with linear models."""
import os

# avoid BLAS thread oversubscription on shared machines (joblib workers already run single-threaded); override by exporting the variables
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
