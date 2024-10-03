import logging

import numpy as np
import scipy

import sklearn.metrics

# for leiden clustering
import sklearn.neighbors
import igraph
import leidenalg


def compute_pdist(X, Y = None, metric = 'l1'):
    
    # compute pairwise distance matrix between two sets of samples
    return sklearn.metrics.pairwise_distances(X, Y, metric = metric, n_jobs = -1)


def pointwise_empirical_distance(XX, XY, YY):
    
    # safety check; refactor with quantiles instead of raw sorting if you need to compare asymmetric distributions
    assert len(XX) == len(YY), 'implementation requires equal sample size'

    # return the pointwise empirical distance from the distance matrices
    return (np.mean(np.abs(np.sort(XY) - np.sort(XX))) + np.mean(np.abs(np.sort(XY.T) - np.sort(YY)))) / 2


def pointwise_empirical_divergence(XX, XY):
    
    # safety check; refactor with quantiles instead of raw sorting if you need to compare asymmetric distributions
    assert len(XX) == len(XY) and len(XX) == len(XY.T), 'implementation requires equal sample size'

    # return the pointwise empirical divergence from the distance matrices
    return np.mean(np.abs(np.sort(XY) - np.sort(XX)))


def energy_distance(XX, XY, YY):

    # return the energy distance from the distance matrices
    return np.sqrt(2 * np.mean(XY) - np.mean(XX) - np.mean(YY))


def null_distribution(X, distance = 'distance', metric = 'l1', n_bootstrap = 100):
    
    # bootstrap pointwise empirical distances of an empirical distribution with respect to itself;
    # return the resulting theoretical distribution
    assert distance in ['distance', 'divergence', 'energy distance']
    
    # compute the internal pairwise distance to build the distribution
    pdist = X if metric == 'precomputed' else compute_pdist(X)
    
    # gather empirical PEDs
    distances = []
    for i in range(n_bootstrap):
        
        Ai = np.random.choice(len(X), len(X))
        Bi = np.random.choice(len(X), len(X))

        if distance == 'divergence':

            # compute the pointwise empirical divergence between the two empirical distributions
            dist = pointwise_empirical_divergence(pdist[np.ix_(Ai, Ai)], pdist[np.ix_(Ai, Bi)])

        elif distance == 'distance':

            # compute the pointwise empirical distance between the two empirical distributions
            dist = pointwise_empirical_distance(pdist[np.ix_(Ai, Ai)], pdist[np.ix_(Ai, Bi)], pdist[np.ix_(Bi, Bi)])

        elif distance == 'energy distance':

            # compute the pointwise empirical distance between the two empirical distributions
            dist = energy_distance(pdist[np.ix_(Ai, Ai)], pdist[np.ix_(Ai, Bi)], pdist[np.ix_(Bi, Bi)])
            
        distances.append(dist)

    # approximate the distributions with a gamma and return
    return scipy.stats.gamma(*scipy.stats.gamma.fit(distances))


def cluster_with_leiden(X, resolution=1, knn=15, distance_metric='l1'):
    
    # either copy distance metric (X) if precomputed, otherwise compute it from the inputs
    distance_matrix = np.copy(X) if distance_metric == 'precomputed' else compute_pdist(X, metric = distance_metric)
        
    # construct connectivity matrix
    connectivity_matrix = sklearn.neighbors.kneighbors_graph(distance_matrix, metric = 'precomputed', n_neighbors = knn, mode = 'connectivity').astype(bool)
    
    # convert to igraph for leiden
    graph = igraph.Graph(n=len(distance_matrix), edges=list(zip(*connectivity_matrix.nonzero())), directed = False)
    
    # cluster with leiden
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter = resolution)
    
    # return labels
    return np.array(partition.membership)


def get_centroids_repeats(pdist, labels):

    # store each cluster medioid, computed as the point minimizing the median distance between all others in the cluster
    centroids_repeats = np.zeros(len(pdist)).astype(int)

    for label in np.unique(labels):

        cluster_members = np.where(labels == label)[0]

        # median distance of cluster members to each others
        median_dist = np.median(pdist[np.ix_(cluster_members, cluster_members)], axis=0)

        # attribute to the centroid an amount of replicates equal to the cluster numerosity
        centroids_repeats[cluster_members[np.argmin(median_dist)]] = len(cluster_members)

    return centroids_repeats


def get_clusters_pairings(centroids_repeats, labels, global_scope, local_scope):
    
    # store the index pairing of all points in the reference set to the ones in the evaluated set for the local scoring
    # store centroids-induced subsets for balanced/local scoring
    subsets, cum_i = [], 0

    if global_scope:
        
        # if global scope is active, the first subset will be the global one
        subsets.append((np.arange(len(centroids_repeats)), np.arange(len(centroids_repeats))))

    if local_scope:
        
        # if local scope is active, we 
        for i in np.where(centroids_repeats>0)[0]:

            # closest points in reference group to current centroid
            reference_i = np.where(labels==labels[i])[0]

            # current points (in initial_points)
            points_i = np.arange(cum_i,cum_i+centroids_repeats[i])

            # append indices of current subset in pairwise distance matrix
            subsets.append((reference_i, points_i))

            # update cumulative index to keep track of where we will be in the sampled population
            cum_i += centroids_repeats[i]

    return subsets


class CoverageEstimator:
    
    '''
    The coverage estimator is the most important piece of the scoring pipeline
    It's initialized at every scoring round, and takes as input a set of reference points that
    will then be used to check how the simulated data is moving through the reference manifold
    '''
    
    def __init__(
        self, validator = None, validity_penalty_exp = 2, approx_p = 1.0, scope = 'balanced', distance_type = 'divergence', distance_metric = 'l1', clustering_resolution = 1.0, local_knn = 15, min_dist_q = None, bootstrap_n = 100):
        
        # validate multiple choice options
        assert scope in ['balanced', 'global', 'local', 'equal'], 'unrecognized scope'
        assert distance_type in ['divergence', 'distance', 'energy distance'], 'unrecognized distance type'
        
        # store scope for scoring
        self.global_scope = scope in ['balanced', 'global']
        self.local_scope = scope in ['balanced', 'local', 'equal']
        self.scope = scope
        
        # incorporate data validator for ease of use
        self.validator = validator
        
        # if the validity penalty should be linearly proportional to the number of invalid samples or more strict
        self.validity_penalty_exp = validity_penalty_exp
        
        # approx scoring using a percentage of the input dataset
        # (speed decreases with square of samples due to pairwise distance computation with reference set)
        self.approx_p = approx_p
        
        # metric space for pairwise distances computation and distance type
        self.distance_metric = distance_metric
        self.distance_type = distance_type
        
        self.clustering_resolution = clustering_resolution
        self.local_knn = local_knn
        
        # if we want to consider the minimum achievable distance from a sample distribution = 0
        # or if we should estimate the likely minimum distance by bootstrapping the samples and gathering the mean distance wrt itself
        self.bootstrap_n = bootstrap_n
        self.min_dist_q = min_dist_q
        
        self.internal_distributions = None
        self.subsets = None
        
    def fit(self, X):
        
        # select uniform subsample from total population
        self.reference_points_idx = np.random.choice(len(X), int(len(X) * self.approx_p), replace = False)
        self.reference_points = np.copy(X[ self.reference_points_idx ])
        
        # compute reference and initial points distances distribution, then make sure diagonal elements are zero
        # note: we always compute the full pairwise distance for the reference distribution to compute the local clusters,
        # however this could be trivially optimized (and it should) to work with large datasets
        self.reference_pdist = compute_pdist(self.reference_points, metric = self.distance_metric)
        np.fill_diagonal(self.reference_pdist, 0)
        
        # initialize clusters, centroids and subsets idx mappings
        self.initialize_local_subsets()
        
        # create n duplicates of each centroid, depending on the cluster size
        self.initial_points = np.repeat(self.reference_points, self.centroids_repeats, axis = 0)
        
        # eventually initialize internal distributions
        if self.min_dist_q is not None:
            self.internal_distributions = self.estimate_null_distributions()
        
        # fetch local minimum or set to zero for fast execution
        self.min_dist = np.array([ d.ppf(self.min_dist_q) for d in self.internal_distributions ]) if self.min_dist_q is not None else np.zeros(len(self.subsets))
        
        # compute distances of initial population wrt reference points to serve as scoring baseline
        self.initial_distance = self.compute_distance(self.initial_points)
        
    def initialize_local_subsets(self):
        
        # cluster the reference dataset with leiden
        self.labels = cluster_with_leiden(self.reference_pdist, resolution = self.clustering_resolution, knn = self.local_knn, distance_metric = 'precomputed')
        
        # gather centroids
        self.centroids_repeats = get_centroids_repeats(self.reference_pdist, self.labels)
        
        # initialize subsets according to centroids distribution and scoring scope
        self.subsets = get_clusters_pairings(self.centroids_repeats, self.labels, self.global_scope, self.local_scope)
        
        if self.local_scope == 'equal':
            
            # each subset has equal weight
            self.subsets_weights = np.ones(len(self.subsets)) / len(self.subsets)
            
        else:

            # weight each subset by its numerosity
            self.subsets_weights = np.array([ len(reference_ix) for reference_ix, evaluated_ix in self.subsets ]).astype(float)
            self.subsets_weights /= sum(self.subsets_weights)
    
    def compute_validity(self, X):
        
        # complete scoring pipeline relies on validator to penalize samples that have fallen outside of target biology
        return (np.ones(len(X)) if self.validator is None else self.validator(X)).astype(bool)
    
    def estimate_null_distributions(self):
        
        # compute the null distributions for every subset
        return [
            null_distribution(self.reference_pdist[np.ix_(reference_ix, reference_ix)], distance = self.distance_type, metric = 'precomputed', n_bootstrap = self.bootstrap_n)
            for reference_ix, evaluated_ix in self.subsets
        ]
        
    def set_min_q(self, q):
        
        # compute the local null distributions if they haven't been yet
        if self.internal_distributions is None:
            self.internal_distributions = self.estimate_null_distributions()
            
        # set min_dist_q for reference
        self.min_dist_q = q
        
        # set the minimum distance to the reference quantile
        self.min_dist = np.array([ d.ppf(self.min_dist_q) for d in self.internal_distributions ])

    def compute_distance(self, Y):
        
        # note: for evaluating a model on the local scope we don't need to compute the full pairwise distance and could simply do something like
        # pdist[np.ix_(rix, eix)] = pairwise_distances(reference_points[rix], Y[eix]) for rix, eix in subsets
        # however the largely inefficient python implementation of this class would overshadow any marginal gains obtained on large datasets
        
        # for pointwise empirical divergence we don't need YY in downstream estimates
        if self.distance_type != 'divergence':
            YY = compute_pdist(Y, metric = self.distance_metric)
        
        # compute distances distribution of all sample points wrt the reference ones
        XY = compute_pdist(self.reference_points, Y, metric = self.distance_metric)
        
        # compute valid points to penalize score (if validator is provided)
        validity_mask = self.compute_validity(Y)
        
        # initialize distances for every distribution cluster
        distances = np.zeros(len(self.subsets))
        
        for subset_i in range(len(self.subsets)):
            
            reference_points, sampled_points = self.subsets[subset_i]
            
            # pairwise distances within the reference points considered in the subset
            xx = self.reference_pdist[np.ix_(reference_points, reference_points)]
            
            # pairwise distances between the reference points considered in the subset and the relative evaluated points
            xy = XY[np.ix_(reference_points, sampled_points)]
            
            if self.distance_type == 'divergence':

                # compute the pointwise empirical divergence between the two empirical distributions
                distance = pointwise_empirical_divergence(xx, xy)
            
            elif self.distance_type == 'distance':

                # compute the pointwise empirical distance between the two empirical distributions
                distance = pointwise_empirical_distance(xx, xy, YY[np.ix_(sampled_points, sampled_points)])
            
            elif self.distance_type == 'energy distance':

                # compute the pointwise empirical distance between the two empirical distributions
                distance = energy_distance(xx, xy, YY[np.ix_(sampled_points, sampled_points)])
            
            # compute validity penalty in case it exists
            validity_penalty = ((len(sampled_points)+1) / (sum(validity_mask[sampled_points])+1))**self.validity_penalty_exp
            
            # compute the distance, inversely scaled by the portion of invalid points and lower bound at the subset min distance
            distances[subset_i] = np.maximum(0, distance * validity_penalty - self.min_dist[subset_i])
        
        # return distances for each cluster
        return distances
    
    def score_raw(self, X):
        
        # compute distance within every cluster constellation
        distances = self.compute_distance(X)
        
        # iterate scores over list (subsets can be of different sizes)
        return np.array([ 1 - distances[i] / self.initial_distance[i] for i in range(len(self.initial_distance)) ])
    
    def score(self, X, elu_score = True, aggregate = True):
        
        # eventually apply elu to clusters score, bounding negative distance score to (-1, 0)
        elu = lambda x: np.where(x > 0, x, np.exp(x) - 1)
        
        # score each cluster and aggregate local scores
        scores = self.score_raw(X)
        
        # eventually apply elu to clusters score, bounding negative distance scores to (-1, 0)
        if elu_score:
            scores = elu(scores)
            
        # eventually aggregate weighting each cluster by its numerosity
        return sum(scores * self.subsets_weights) if aggregate else scores