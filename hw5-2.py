#!/usr/bin/env python3

import numpy as np
import csv, math
import matplotlib.pyplot as plt

def parse(filename):
    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            data.append(list(map(float,row)))
    return np.array(data)

def buggy_PCA(X,d):
    """
    Buggy PCA in d dimensions
    Given data as rows in X, returns Vt and X1 (reconstruction)
    """
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    Vtd = Vt[:d,:]
    X1 = (Vtd.T @ Vtd @ X.T).T
    return Vt[:d,:], X1

def demeaned_PCA(X,d):
    """
    De-meaned PCA in d dimensions
    Given data as rows in X, shifts rows until mean is 0,
    then returns Vt, X1, xbar
    """
    xbar = np.average(X, axis=0)
    X = X - xbar
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    Vtd = Vt[:d,:]
    X1 = (Vtd.T @ Vtd @ X.T).T + xbar
    return Vt[:d,:], X1, xbar

def normalize(X):
    """
    Normalizes X such that mean is 0 and sd in each dimension is 1
    """
    xbar = np.average(X, axis=0)
    X = X - xbar
    n = np.shape(X)[1]
    sd = np.array([np.std(X[:,i]) for i in range(n)])
    for i in range(n):
        X[:,i] /= sd[i]
    return X, xbar, sd

def denormalize(X, xbar, sd):
    """
    Given normalized X and xbar, sd of previous X, restores X
    """
    n = np.shape(X)[1]
    for i in range(n):
        X[:,i] *= sd[i]
    X = X + xbar
    return X

def normalized_PCA(X,d):
    """
    Given data as rows in X, changes rows until mean is 0,
    and sd of each dimension is 1, then returns A
    """
    X, xbar, sd = normalize(X)
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    Vtd = Vt[:d,:]
    X1 = denormalize((Vtd.T @ Vtd @ X.T).T, xbar, sd)
    return Vt[:d,:], X1, xbar, sd

def DRO(X, d):
    """
    Given data as rows in X, returns Z, At, b for DRO representation
    on d dimensions, along with reconstruction
    """
    n = np.shape(X)[0]
    xbar = np.average(X, axis=0)
    Y = X - xbar
    U, Sigma, Vt = np.linalg.svd(Y)
    U1 = U[:,:d]
    Sigma1 = Sigma[:d]
    V1 = (Vt.T)[:,:d]

    Z = math.sqrt(n) * U1
    A = (V1 @ np.diag(Sigma1).T) / math.sqrt(n)
    X1 = (Z @ A.T) + xbar
    return Z, A, xbar, X1

def recon_error(X1, X):
    """
    Returns the total reconstruction error of X1 compared to X
    """
    error = 0.0
    for i in range(np.shape(X)[0]):
        error += math.pow(np.linalg.norm(X1[i] - X[i]), 2)
    return error

def p2_2D():
    print("data2D.csv report:")
    filename = 'data/data2D.csv'
    X = parse(filename)
    # Buggy PCA
    buggy_Vt, buggy_X = buggy_PCA(X,1)
    print('Reconstruction Error for Buggy PCA:')
    print(recon_error(buggy_X, X))
    plt.scatter(X[:,0], X[:,1], color='blue', \
                label='Original Data', marker='o')
    plt.scatter(buggy_X[:,0], buggy_X[:,1], color='red',\
                label='Buggy Reconstruction', marker='x')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.title('Buggy PCA Reconstruction')
    plt.legend()
    plt.show()

    # De-meaned PCA
    dem_Vt, dem_X, xbar = demeaned_PCA(X,1)
    print('Reconstruction Error for De-meaned PCA:')
    print(recon_error(dem_X, X))
    plt.scatter(X[:,0], X[:,1], color='blue', \
                label='Original Data', marker='o')
    plt.scatter(dem_X[:,0], dem_X[:,1], color='red',\
                label='De-meaned Reconstruction', marker='x')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.title('De-meaned PCA Reconstruction')
    plt.legend()
    plt.show()

    # Normalized PCA
    norm_Vt, norm_X, xbar, sd = normalized_PCA(X,1)
    print('Reconstruction Error for Normalized PCA:')
    print(recon_error(norm_X, X))
    plt.scatter(X[:,0], X[:,1], color='blue', \
                label='Original Data', marker='o')
    plt.scatter(norm_X[:,0], norm_X[:,1], color='red',\
                label='Normalized Reconstruction', marker='x')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.title('Normalized PCA Reconstruction')
    plt.legend()
    plt.show()

    # DRO
    DRO_Z, DRO_A, DRO_b, DRO_X = DRO(X, 1)
    print('Reconstruction Error for DRO:')
    print(recon_error(DRO_X, X))
    plt.scatter(X[:,0], X[:,1], color='blue', \
                label='Original Data', marker='o')
    plt.scatter(DRO_X[:,0], DRO_X[:,1], color='red',\
                label='DRO Reconstruction', marker='x')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.title('DRO Reconstruction')
    plt.legend()
    plt.show()

def p2_1000D():
    print("data1000D.csv report:")
    filename = 'data/data1000D.csv'
    X = parse(filename)

    # Singular value analysis
    xbar = np.average(X, axis=0)
    Y = X - xbar
    U, Sigma, Vt = np.linalg.svd(Y)

    plt.scatter(range(len(Sigma)), Sigma, color='blue')
    plt.xlim(0,len(Sigma))
    plt.ylim(0,1000)
    plt.title('Singular Values of Centered 1000D Data')
    plt.show()

    # Cutoff value is roughly 500
    d = len([v for v in Sigma if v > 500])
    # d = 30
    print('d = {:d}'.format(d))

    # Buggy PCA
    buggy_Vt, buggy_X = buggy_PCA(X,d)
    print('Reconstruction Error for Buggy PCA:')
    print(recon_error(buggy_X, X))

    # De-meaned PCA
    dem_Vt, dem_X, xbar = demeaned_PCA(X,d)
    print('Reconstruction Error for De-meaned PCA:')
    print(recon_error(dem_X, X))

    # Normalized PCA
    norm_Vt, norm_X, xbar, sd = normalized_PCA(X,d)
    print('Reconstruction Error for Normalized PCA:')
    print(recon_error(norm_X, X))

    # DRO
    DRO_Z, DRO_A, DRO_b, DRO_X = DRO(X,d)
    print('Reconstruction Error for DRO:')
    print(recon_error(DRO_X, X))

p2_1000D()
