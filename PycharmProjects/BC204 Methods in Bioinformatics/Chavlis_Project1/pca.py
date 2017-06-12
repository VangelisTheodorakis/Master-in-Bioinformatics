import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose as tr

from sklearn.decomposition import PCA


def do_PCA(x_array, y):
    do_plot(x_array, y,"Dataset")
    hat_x_array = np.empty([x_array.shape[0], x_array.shape[1]])

    for data_column in range(0, x_array.shape[1]):
        print("peos")


        print("column ", x_array[:, data_column])
        print("mean ", np.mean(x_array[:, data_column]))
        hat_x_array[:, data_column] = np.mean(x_array[:, data_column])

    print("hat x is %s ", hat_x_array)
    normalized_x = np.subtract(x_array, hat_x_array)
    print("x array", x_array)
    print("normalized x %s", normalized_x)

    #covarianve_matrix = (1/x_array.shape[1]) * normalized_x.dot(tr(normalized_x))
    covarianve_matrix = np.cov(x_array)

    print("covariance ", covarianve_matrix)
    print("conv ", np.cov(normalized_x))
    eigenvalues, eigenvectors = np.linalg.eig(covarianve_matrix)

    for i in range(len(eigenvalues)):
        eigv = eigenvectors[:, i].T
        np.testing.assert_array_almost_equal(covarianve_matrix.dot(eigv), eigenvalues[i] * eigv,decimal=6, err_msg='', verbose=True)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # sort eigenvectors according to same index
    eigenvalues = eigenvalues[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    print("--->",eigenvectors.shape)
    eigenvectors = eigenvectors[:2, :2]
    print("--->", eigenvectors.shape)
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    
    projections = eigenvectors.dot(tr(x_array))

    print("projection ", projections.shape)

    do_plot(projections.T, y,  "My_Custom_PCA")
    validate(x_array, y)


def do_plot(data, y, title):
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


def validate(x_array, y):
        pca = PCA()
        standard_pca = pca.fit_transform(x_array)
        do_plot(standard_pca, y, "Default_PCA")
