from pca import *
from kpca import *
from ppca import *

from sklearn.datasets import make_circles
import os

def prepare_data():
    if not os.path.exists("204_mice_data.txt"):
        file = os.system("wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz")
        os.system('gunzip GDS6248.soft')
        os.system('grep -i ILMN GDS6248.soft > Data.txt')
        os.system('cut -f3- Data.txt > 204_mice_data.txt')

    data = np.loadtxt("204_mice_data.txt")
    return data

X, y = make_circles(n_samples=1000, factor=.3, noise=.05)

# PCA

# ============================================================

do_PCA(X, y)

# ============================================================

# KPCA

do_KPCA(X, y)

do_with_variant_kernels(X, y, "rbf", "Gaussian Kernel")
do_with_variant_kernels(X, y, "poly", "Polynomial Kernel")


# ============================================================

# PPCA with circles

do_PPCA(X, y, 2, np.random.rand(), 20, 1)

# PPCA with real data

do_PPCA(prepare_data(), y,2, np.random.rand(), 20, 0)