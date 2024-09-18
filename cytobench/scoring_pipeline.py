import logging
import numpy as np

from .coverage_estimator import CoverageEstimator

def score_model(
    sampling_fn, environment_points, data_validator = None, validity_penalty_exp = 1, aggregate_subclusters = True, approx_scoring = False, 
    scope = 'balanced', elu_score = True, n_iterations = 1000, plateau_after_n = None, score_every_n = 1, 
    min_dist_q = None, bootstrap_n = 30, approx_p = 1.0, clustering_resolution = 1.0, local_knn = 15, distance_metric='l1'):
    
    ''' 
    Fully score a single trajectory (n of these will make up the actual score interval for a model)
    If approx_scoring = True we only score once every plateau_after_n epochs, then we go back and try to find the
    max score scoring intermediate points
    '''
    
    # initialize coverage estimator
    coverage_estimator = CoverageEstimator(
        validator = data_validator, validity_penalty_exp = validity_penalty_exp, approx_p = approx_p, scope = scope, 
        clustering_resolution = clustering_resolution, local_knn = local_knn, 
        min_dist_q = min_dist_q, bootstrap_n = bootstrap_n, 
        distance_metric = distance_metric
    )
    coverage_estimator.fit(environment_points)
    
    # extract initial population
    current_population = np.copy(coverage_estimator.initial_points)

    # initialize cycle variables 
    scores, clusters_scores, populations, population_size = [], [], [np.copy(current_population)], len(current_population)
    
    # scoring step is determined by approx scoring
    scoring_step = plateau_after_n if approx_scoring else score_every_n

    # start sampling
    for n_iteration in range(n_iterations):

        # actually score the model only once every score_every_n epochs, (and always in the last iteration)
        # note: the first iteration should and will always be scored for the baseline
        if n_iteration % scoring_step == 0 or n_iteration == n_iterations - 1:
            
            # store stats
            clusters_scores.append(coverage_estimator.score(current_population, elu_score=elu_score, aggregate=False))
            scores.append(sum(clusters_scores[-1] * coverage_estimator.subsets_weights))

            # interrupt in case of plateau (if min_epochs has been reached)
            # using > plateau_after_n (rather than >=) will ensure that we extend for one extra cycle, which is needed for the part that comes after
            if plateau_after_n is not None and n_iteration - np.nanargmax(scores) > plateau_after_n:
                break
                
            # remove sampled populations that for sure we won't score to free up memory for long runs
            # the populations to keep are the ones within plateau_after_n radius of the plateau epoch and
            # from the endcap, since a new max might be coming; we do this but simply iterating over the
            # whole array to keep the code clean
            if approx_scoring:
                for i in range(len(populations)):
                    if populations[i] is not None:
                        if np.abs(i - np.nanargmax(scores)) > plateau_after_n and np.abs(i - n_iteration) > plateau_after_n:
                            populations[i] = None
            
        else:
            scores.append(np.nan)
            clusters_scores.append([np.nan] * len(clusters_scores[0]))
            
        # sample new population at every epoch
        current_population = sampling_fn(current_population)
        
        # store the sampled population if approx scoring is active (and we might need it for later evaluation)
        populations.append(np.copy(current_population) if approx_scoring else None)

        # make sure that the sampling function didn't drop any point
        assert len(current_population) == population_size, "Sampling function seems to have dropped some points"
        
        # note: the max value of reads should be << 100, if it goes over 1000 the sampling has become unstable and will likely crash the pipeline
        if np.any(current_population > 1000):
            logging.warning(f'Sampled values are diverging: max = {np.max(current_population)}, note that the maximum count value should be << 100')
            break
        
    # check the indices of the first not None value before and after the plateau epoch
    ia, ib = get_non_nan_extremes(scores, np.nanargmax(scores))
    
    # compute scores of intermediate values until the desired granularity has been reached
    while max(np.nanargmax(scores) - ia, ib - np.nanargmax(scores)) > score_every_n:
        
        # consider intermediate point between last not None and the plateau epoch from left and right
        ia = np.ceil((np.nanargmax(scores) + ia) / 2).astype(int)
        ib = np.floor((np.nanargmax(scores) + ib) / 2).astype(int)
        
        for i in [ia, ib]:

            if np.isnan(scores[i]):
                
                # store stats
                clusters_scores[i] = coverage_estimator.score(populations[i], elu_score=elu_score, aggregate=False)
                scores[i] = sum(clusters_scores[i] * coverage_estimator.subsets_weights)
                    
        # update the empty indices and iterate
        ia, ib = get_non_nan_extremes(scores, np.nanargmax(scores))

    # cast scores to numpy array, then interpolate nan elements before returning
    clusters_scores, scores, nans, indices = np.array(clusters_scores).T, np.array(scores), np.isnan(scores), np.arange(len(scores))
        
    # interpolate using np.interp for indices where scores is nan
    scores[nans] = np.interp(indices[nans], indices[~nans], scores[~nans])
    
    # repeat for relative scores
    for i in range(clusters_scores.shape[0]):
        clusters_scores[i][nans] = np.interp(indices[nans], indices[~nans], clusters_scores[i][~nans])
        
    # either return single score for each epoch, or relative one of each cluster
    return scores if aggregate_subclusters else (clusters_scores, coverage_estimator.subsets_weights)


def get_non_nan_extremes(v, i):
    
    '''
    Given a list of values with some elements == None, finds and returns the indices of the closest non nans value to its left and right
    (eventually returning itself if i==0 for the left value)
    Note: the algorithm is based on the assumption that i is not None and won't yield correct results otherwise
    '''

    # collect np.nan indices
    nans = np.isnan(v)

    # indices for interpolation
    indices = np.arange(len(v))
    
    # left index
    left_indices = indices[:i+1][~nans[:i+1]]
    ia = left_indices[-2:][0]
    
    # right index
    right_indices = indices[i:][~nans[i:]]
    ib = right_indices[:2][-1]

    return ia, ib