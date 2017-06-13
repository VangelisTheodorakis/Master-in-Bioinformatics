import numpy as np
import pylab as plt


def fit_EM( X, k,max_iters=1000, reps = 10, eps=0.000001):

    print("fit em")
    # n = number of data-points, d = dimension of data points

    n, d =X.shape

    print("Rows ",X.shape[0]," x Columns", X.shape[1])

    # randomly choose the starting centroids/means
    ## as k of the points from datasets

    mu = X[np.random.choice(n, k, False), :]

    print("Mu shape", mu.shape, " mu", mu)
    # initialize the covariance matrices for each gaussians
    Sigma = [np.eye(d)] * k

    # initialize the probabilities/weights for each gaussians
    w = [1. / k] * k

    # responsibility matrix is initialized to all zeros
    # we have responsibility for each of n points for eack of k gaussians
    R = np.zeros((n, k))
    ### log_likelihoods
    log_likelihoods = []

    P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) \
                      * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                               X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

    # Iterate till max_iters iterations
    while len(log_likelihoods) < max_iters:

        # E - Step

        ## Vectorized implementation of e-step equation to calculate the
        ## membership for each of k -gaussians
        for k in range(k):
            R[:, k] = w[k] * P(mu[k], Sigma[k])

        ### Likelihood computation
        log_likelihood = np.sum(np.log(np.sum(R, axis=1)))

        log_likelihoods.append(log_likelihood)

        ## Normalize so that the responsibility matrix is row stochastic
        R = (R.T / np.sum(R, axis=1)).T

        ## The number of datapoints belonging to each gaussian
        N_ks = np.sum(R, axis=0)

        # M Step
        ## calculate the new mean and covariance for each gaussian by
        ## utilizing the new responsibilities
        for k in range(k):
            ## means
            mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis=1).T
            x_mu = np.matrix(X - mu[k])

            ## covariances
            Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, R[:, k]), x_mu))

            ## and finally the probabilities
            w[k] = 1. / n * N_ks[k]
        # check for onvergence
        if len(log_likelihoods) < 2: continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < eps: break

    ## bind all results together
    from collections import namedtuple
    params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
    params.mu = mu
    params.Sigma = Sigma
    params.w = w
    params.log_likelihoods = log_likelihoods
    params.num_iters = len(log_likelihoods)

    return R, params


def my_mog(X, k, maxiter, reps):
    R, params = fit_EM(X, k, maxiter, reps)

    print("Responsibility shape", R.shape, " Responsibility", R)

    show(X, R, 'Mixture of Gaussians')

def show(X, R, title):
    plt.cla()
    plt.title(title)
    print(X.shape)
    if R.shape[1] == 2:
        plt.plot(X[np.argmax(R, axis=1) == 0], 'ob',X[np.argmax(R, axis=1) == 1], 'or')
    elif R.shape[1] == 3:
        plt.plot(X[np.argmax(R, axis=1) == 0], 'ob',
                 X[np.argmax(R, axis=1) == 1], 'or',
                 X[np.argmax(R, axis=1) == 2], 'og')
    plt.draw()
    plt.ioff()
    plt.savefig(title + ".png")
    plt.show()