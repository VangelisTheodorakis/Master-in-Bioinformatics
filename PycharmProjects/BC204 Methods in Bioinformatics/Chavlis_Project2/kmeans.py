import pylab as plt
import numpy as np
from numpy import transpose as tr
plt.ion()


def my_kmeans(X, k, distance, maxiter, reps, practical):
    for replications in range(reps):
        # Randomly pick centroids
        centroids = X[np.random.choice(np.arange(len(X)), k), :]
        # print("centroids",centroids)
        for i in range(maxiter):
            # Cluster Assignment step
            # Choose Method for clustering
            if distance == 'euclidean':
                C = np.array([np.argmin([np.sqrt(np.dot(x_i - y_k, x_i - y_k)) for y_k in centroids]) for x_i in X])
            elif distance == 'mahalanobis':
                C = np.array([np.argmin([np.sqrt(tr(x_i - y_k).dot(np.cov(x_i - y_k)).dot(x_i - y_k)) for y_k in centroids]) for x_i in X])
            elif distance == 'manhattan':
                C = np.array([np.argmin([np.absolute(x_i - y_k) for y_k in centroids]) for x_i in X])
            else:
                print("Wrong Distance!")
                return
            # Move centroids step
            centroids = [X[C == k].mean(axis=0) for k in range(k)]
    if(practical == 1):
        show(X, C, np.array(centroids), distance+" practical")
    else:
        show(X, C, np.array(centroids), distance)
    return


def show(X, C, centroids, title):
    plt.cla()
    plt.title(title)
    print(X.shape)
    if np.amax(C) == 1:
        plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
             X[C == 1, 0], X[C == 1, 1], 'or')
    elif np.amax(C) == 2:
        plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
                 X[C == 1, 0], X[C == 1, 1], 'or',
                 X[C == 2, 0], X[C == 2, 1], 'og')
    elif np.amax(C) == 3:
        plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
                 X[C == 1, 0], X[C == 1, 1], 'or',
                 X[C == 2, 0], X[C == 2, 1], 'og',
                 X[C == 3, 0], X[C == 3, 1], 'oy')
    elif np.amax(C) == 4:
        plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
                 X[C == 1, 0], X[C == 1, 1], 'or',
                 X[C == 2, 0], X[C == 2, 1], 'og',
                 X[C == 3, 0], X[C == 3, 1], 'og',
                 X[C == 4, 0], X[C == 4, 1], 'op')
    plt.plot(centroids[:, 0], centroids[:, 1], '*m', markersize=20)
    plt.draw()
    plt.ioff()
    plt.savefig(title + ".png")
    plt.show()