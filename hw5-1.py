#!/usr/bin/env python3

import numpy as np
import itertools, math, scipy
import matplotlib.pyplot as plt

DISTS = [{'mean': np.array([-1,-1]),
          'cov': np.array([[2,0.5],[0.5,1]])},
         {'mean': np.array([1,-1]),
          'cov': np.array([[1,-0.5],[-0.5,2]])},
         {'mean': np.array([0,1]),
          'cov': np.array([[1,0],[0,2]])}]

NUM_CENTERS = 3
NUM_POINTS = 100 # per distribution

def eval_correct(labels, predicted):
    """
    Evaluates accuracy of predicted labels
    given that predicted labels are a permutation of original labels
    This would be really inefficient for a large number of centers.
    """
    correct = 0
    for perm in itertools.permutations(range(NUM_CENTERS)):
        correct1 = 0
        for i in range(len(labels)):
            if labels[i] == perm[predicted[i]]:
                correct1 += 1
        correct = max(correct, correct1)
    return correct/len(labels)

def centers_init(dataset):
    """
    Basic initial center selection
    Just chooses random points in the dataset
    """
    indices = np.random.choice(np.shape(dataset)[0],NUM_CENTERS)
    return np.array([dataset[i] for i in indices])

def kmeanspp(dataset):
    """
    kmeans++ method of initial center selection
    """
    pass

def kmeans_centers(dataset, init_fn=centers_init):
    """
    Returns the k-means centers and labels from
    one run of k-means clustering
    """
    n = np.shape(dataset)[0]
    centers = init_fn(dataset)
    labels = np.full(n, 0)
    while True:
        # assign points to centers
        new_labels = np.full(n, 0)
        for i in range(n):
            distances = np.array([np.linalg.norm(c-dataset[i]) \
                                  for c in centers])
            new_labels[i] = np.argmin(distances)
        # if k-means done (labels don't change)
        if np.array_equal(new_labels,labels):
            return centers, labels
        # else compute new centers
        labels = new_labels
        for c in range(NUM_CENTERS):
            sum = np.array([0.0,0.0])
            cnt = 0
            for i in range(n):
                if labels[i] == c:
                    sum += dataset[i]
                    cnt += 1
            if cnt > 0:
                centers[c] = sum/cnt
                # else don't update

def kmeans_eval(dataset, labels, runs=15, init_fn=centers_init):
    """
    Evaluates the objective function and accuracy of k-means
    on previously labeled data
    This takes multiple runs and selects the run with lowest objective
    Output:
    - centers: resulting k-means centers
    - objective: value of the objective function on dataset
    - accuracy: proportion of correctly labeled instances (0-1)
    """
    result = { 'centers': None, 'predictions': None, 'objective': None,
               'accuracy': None }
    for _ in range(runs):
        centers, predicted_labels = kmeans_centers(dataset, init_fn)
        # compute objective
        objective = 0.0
        for i in range(np.shape(dataset)[0]):
            objective += np.linalg.norm(dataset[i] \
                                        - centers[predicted_labels[i]])
        # compute accuracy
        correct = eval_correct(labels, predicted_labels)
        # update result if objective is better
        if result['objective'] is None or objective < result['objective']:
            result = {'centers': centers,
                      'predictions': predicted_labels,
                      'objective': objective,
                      'accuracy': correct}
    return result

def gmm_centers(dataset, init_fn=centers_init):
    """
    Returns the result of one run of Gaussian mixture model
    on a dataset: means, covariances, weights (phi)
    """
    # Initialize
    n = np.shape(dataset)[0]
    means = init_fn(dataset)
    covs = np.full((NUM_CENTERS,2,2),np.cov(dataset.T)) # start with sample covariance
    phi = np.full(NUM_CENTERS, 1/NUM_CENTERS)
    labels = np.full(n,0) # this is just to check stopping
    w = np.empty((n,NUM_CENTERS))
    iters = 0
    while iters < 50:
        iters += 1
        # E step
        new_labels = np.full(n,0)
        for i in range(n):
            for j in range(NUM_CENTERS):
                normal = scipy.stats.multivariate_normal(means[j],covs[j])
                w[i][j] = normal.pdf(dataset[i]) * phi[j]
            w[i] /= sum(w[i]) # Normalize

        # M step
        for j in range(NUM_CENTERS):
            phi[j] = np.average(w[:,j])
            means[j] = np.average(dataset, axis=0, weights=w[:,j])
            covs[j] = np.array([[0,0],[0,0]])
            for i in range(n):
                covs[j] += w[i][j] * np.outer(dataset[i]-means[j],dataset[i]-means[j])
            covs[j] /= np.sum(w[:,j])
    return means, covs, phi, w

def gmm_eval(dataset, labels, runs=15, init_fn=centers_init):
    """
    Evaluates the objective function and accuracy of k-means
    on previously labeled data
    This takes multiple runs and selects the run with lowest objective
    Output:
    - centers: resulting GMM centers
    - covs: resulting GMM covariances
    - objective: value of the objective function on dataset
    - accuracy: proportion of correctly labeled instances (0-1)
    """
    result = { 'centers': None, 'covs': None,'predictions': None, 'objective': None,
               'accuracy': None }
    for _ in range(runs):
        centers, covs, phi, w = gmm_centers(dataset, init_fn)
        # compute objective (log likelihood)
        objective = 0.0
        n = np.shape(dataset)[0]
        normals = np.array([scipy.stats.multivariate_normal(centers[j],covs[j]) \
                            for j in range(NUM_CENTERS)])
        for i in range(n):
            objective += math.log2(sum([normals[j].pdf(dataset[i]) \
                                        for j in range(NUM_CENTERS)]))
        # compute accuracy
        # we take the argmax label for highest weight
        # to ensure a fair comparison to k-means
        predicted_labels = np.full(n,0)
        for i in range(n):
            predicted_labels[i] = np.argmax(w[i])
        correct = eval_correct(labels, predicted_labels)
        # update result if objective is better
        if result['objective'] is None or objective < result['objective']:
            result = {'centers': centers,
                      'covs': covs,
                      'predictions': predicted_labels,
                      'objective': objective,
                      'accuracy': correct}
    return result


def p1_step(sigma):
    # Sample dataset
    dataset = np.empty(shape=(len(DISTS)*NUM_POINTS,2))
    labels = np.empty(shape=(len(DISTS)*NUM_POINTS))
    for i in range(len(DISTS)):
        dist = DISTS[i]
        for j in range(NUM_POINTS):
            dataset[i*NUM_POINTS+j] = \
                np.random.multivariate_normal(dist['mean'], sigma*dist['cov'])
            labels[i*NUM_POINTS+j] = i
    print(kmeans_eval(dataset, labels))
    print(gmm_eval(dataset, labels))

def p1():
    sigmas = [0.5, 1, 2, 4, 8]
    kmeans_acc = np.full(len(sigmas),0.0)
    gmm_acc = np.full(len(sigmas),0.0)
    for t in range(len(sigmas)):
        print("Evaluating sigma={}".format(sigmas[t]))
        dataset = np.empty(shape=(len(DISTS)*NUM_POINTS,2))
        labels = np.empty(shape=(len(DISTS)*NUM_POINTS))
        for i in range(len(DISTS)):
            dist = DISTS[i]
            for j in range(NUM_POINTS):
                dataset[i*NUM_POINTS+j] = \
                    np.random.multivariate_normal(dist['mean'], sigmas[t]*dist['cov'])
                labels[i*NUM_POINTS+j] = i
        kmeans_ev = kmeans_eval(dataset,labels)
        print(kmeans_ev)
        gmm_ev = gmm_eval(dataset,labels)
        print(gmm_ev)
        kmeans_acc[t] = kmeans_ev['accuracy']
        gmm_acc[t] = gmm_ev['accuracy']
    plt.plot(sigmas,kmeans_acc,color='blue',label='K-means', marker='o')
    plt.plot(sigmas,gmm_acc,color='red',label='GMM',marker='o')
    plt.xlabel('Sigma')
    plt.ylabel('Accuracy')
    plt.title('Clustering Accuracy')
    plt.legend()
    plt.show()

p1()
