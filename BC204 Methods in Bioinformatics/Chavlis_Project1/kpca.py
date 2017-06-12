from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import KernelPCA

def do_KPCA(X, y, gamma=10, n_components=2):

    do_plot(X, y,"Initial Data")
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    do_plot(X_pc, y,"My_Custom_KPCA")

    return

def do_plot(data, y,title):
    print("data space" , data.shape)
    plt.scatter(data[y == 0, 0], data[y == 0, 1], color='red', marker='^', alpha=0.5, label='Circle_01')
    plt.scatter(data[y == 1, 0], data[y == 1, 1], color='blue', marker='o', alpha=0.5, label='Circle_02')
    plt.grid(True)
    plt.xlabel('Pca_01')
    plt.ylabel('Pca_02')
    plt.legend(numpoints=1, loc='lower right')
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()

def do_with_variant_kernels(data, y, kernel, title):
    kpca = KernelPCA(kernel=kernel, fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(data)
    do_plot(X_kpca, y,title)