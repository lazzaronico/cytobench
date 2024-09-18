import logging

import numpy as np
import scipy

import sklearn.metrics

# for leiden clustering
import sklearn.neighbors
import igraph
import leidenalg


def compute_ped(reference_samples, evaluated_samples, distance_metric = 'l1'):
    
    '''
    The pointwise distributions distance can be computed using any distance between distributions;
    in this notebook we'll use the wasserstein distance
    '''
    
    if distance_metric == 'precomputed':
        dist_AA, dist_AB = reference_samples, evaluated_samples
    else:
        dist_AA = sklearn.metrics.pairwise_distances(reference_samples, metric=distance_metric, n_jobs=-1)
        dist_AB = sklearn.metrics.pairwise_distances(reference_samples, evaluated_samples, metric=distance_metric, n_jobs=-1)

    # rows of the two distance matrix must be equal since they are computed with respect to the same reference samples
    assert len(dist_AA) == len(dist_AB)

    # the final pointwise empirical distance is given by the pairwise comparison of the distances distribution
    # of each point in the reference set wrt its distances distribution against every point in the evaluated set
    return np.mean([
        scipy.stats.wasserstein_distance(dist_AA[i], dist_AB[i]) 
        for i in range(len(dist_AA))
    ])


def ped_null_distribution(X, distance_metric = 'l1', n_bootstrap = 100):
    
    '''
    bootstrap pointwise empirical distances of an empirical distribution with respect to itself;
    return the resulting theoretical distribution
    '''
    
    # pdist is 
    pdist = X if distance_metric == 'precomputed' else sklearn.metrics.pairwise_distances(X, metric=distance_metric, n_jobs=-1)
    
    # gather empirical PEDs
    peds = []
    for i in range(n_bootstrap):
        
        Ai = np.random.choice(len(X), len(X))
        Bi = np.random.choice(len(X), len(X))
        
        peds.append(compute_ped(pdist[np.ix_(Ai, Ai)], pdist[np.ix_(Ai, Bi)], distance_metric = 'precomputed'))

    # these pairwise distance will approximately follow a Gamma distribution
    return scipy.stats.gamma(*scipy.stats.gamma.fit(peds))


def cluster_with_leiden(X, resolution=1, knn=15, distance_metric='l1'):
    
    # either copy distance metric (X) if precomputed, otherwise compute it from the inputs
    distance_matrix = np.copy(X) if distance_metric == 'precomputed' else sklearn.metrics.pairwise_distances(X, metric = distance_metric, n_jobs = -1)
        
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
    
    '''
    Store the index pairing of all points in the reference set to the ones in the evaluated set for the local scoring
    '''

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
    will then be used to check how the simulated data is moving through the known manifold of any
    given dataset in a robust and biologically meaningful manner
    '''
    
    def __init__(
        self, validator = None, validity_penalty_exp = 2, approx_p = 1.0, scope = 'balanced', distance_metric = 'l1', clustering_resolution = 1.0, local_knn = 15, min_dist_q = None, bootstrap_n = 30):
        
        # validate multiple choice options
        assert scope in ['balanced', 'global', 'local', 'equal'], 'unrecognized scope'
        
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
        
        self.distance_metric = distance_metric
        
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
        self.reference_pdist = self.distances_distribution(self.reference_points)
        np.fill_diagonal(self.reference_pdist, 0)
        
        # initialize clusters, centroids and subsets idx mappings
        self.initialize_local_subsets()
        
        # create n duplicates of each centroid, depending on the cluster size
        self.initial_points = np.repeat(self.reference_points, self.centroids_repeats, axis = 0)
        
        # eventually initialize internal distributions
        if self.min_dist_q is not None:
            self.estimate_null_distributions()
        
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

    def distances_distribution(self, X):
        
        # note: for evaluating a model on the local scope we don't need to compute the full pairwise distance and could simply do something like
        # pdist[np.ix_(rix, eix)] = pairwise_distances(reference_points[rix], X[eix]) for rix, eix in subsets
        # however the largely inefficient implementation of this class (and python in this context) would overshadow any marginal gains obtained on large datasets
        # therefore we just use a simple full pairwise distance computation
        return sklearn.metrics.pairwise_distances(self.reference_points, X, metric = self.distance_metric, n_jobs = -1)
    
    def compute_validity(self, X):
        
        # complete scoring pipeline relies on validator to penalize samples that have fallen outside of target biology
        return (np.ones(len(X)) if self.validator is None else self.validator(X)).astype(bool)
    
    def estimate_null_distributions(self):
        
        # compute the null ped distributions for all local subsets
        # (self.internal_distributions will then be a list of scipy distributions)
        self.internal_distributions = [
            ped_null_distribution(self.reference_pdist[np.ix_(reference_ix, reference_ix)], distance_metric = 'precomputed', n_bootstrap = self.bootstrap_n)
            for reference_ix, evaluated_ix in self.subsets
        ]
        
    def set_min_q(self, q):
        
        # compute the local null distributions if they haven't been yet
        if self.internal_distributions is None:
            self.estimate_null_distributions()
            
        # set min_dist_q for reference
        self.min_dist_q = q
        
        # set the minimum distance to the reference quantile
        self.min_dist = np.array([ d.ppf(self.min_dist_q) for d in self.internal_distributions ])

    def compute_distance(self, X):
        
        # compute distances distribution of all sample points wrt the reference ones
        distances_distribution = self.distances_distribution(X)
        
        # compute valid points to penalize score (if validator is provided)
        validity_mask = self.compute_validity(X)
        
        # 
        subsets_pdes = np.zeros(len(self.subsets))
        
        for subset_i in range(len(self.subsets)):
            
            reference_points, sampled_points = self.subsets[subset_i]
            
            # pairwise distances within the reference points considered in the subset
            reference_pdist = self.reference_pdist[np.ix_(reference_points, reference_points)]
            
            # pairwise distances between the reference points considered in the subset and the relative evaluated points
            evaluated_pdist = distances_distribution[np.ix_(reference_points, sampled_points)]
            
            # compute the PED between the two empirical distributions
            dist = compute_ped(reference_pdist, evaluated_pdist, distance_metric = 'precomputed')
            
            # compute validity penalty in case it exists
            validity_penalty = ((len(sampled_points)+1) / (sum(validity_mask[sampled_points])+1))**self.validity_penalty_exp
            
            # compute the distance, inversely scaled by the portion of invalid points
            subsets_pdes[subset_i] = np.maximum(0, dist * validity_penalty - self.min_dist[subset_i])
        
        # return distances distribution for each cluster;
        # since the numerosity will usually differ between clusters, we don't cast this to numpy array
        return subsets_pdes
    
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