from kmeans import *
from mog import *
from ppca import *
import os

def prepare_data():
    if not os.path.exists("204_mice_data.txt"):
        file = os.system("wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz")
        os.system('gunzip GDS6248.soft')
        os.system('grep -i ILMN GDS6248.soft > Data.txt')
        os.system('cut -f3- Data.txt > 204_mice_data.txt')

    data = np.loadtxt("204_mice_data.txt")
    return data


m1, cov1 = [1, 1], [[0.5, 0], [0, 0.5]]
m2, cov2 = [-1, -1], [[0.75, 0], [0, 0.75]]

data1 = np.random.multivariate_normal(m1, cov1, 220)
data2 = np.random.multivariate_normal(m2, cov2, 280)

data = np.concatenate((data1, data2))

clusters = 2


my_kmeans(data, clusters, 'euclidean', 100, 1, 0)

my_kmeans(data, clusters, 'mahalanobis', 100, 1, 0)

my_kmeans(data, clusters, 'manhattan', 100, 1, 0)



my_mog(data, clusters, 100, 1)





##########################################################
###     Practical
##########################################################

clusters = 2

data = do_PPCA(prepare_data(), 0, 2, np.random.rand(), 20, 0)

my_kmeans(data.T, clusters, 'euclidean', 100, 1, practical = 1)

my_kmeans(data.T, clusters, 'mahalanobis', 100, 1,practical = 1)

my_kmeans(data.T, clusters, 'manhattan', 100, 1, practical = 1)

