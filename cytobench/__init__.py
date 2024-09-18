'''
NOTE: all functions in the codebase have been implemented for demonstration porpose only, without any sanity check on the input data;
      do not use this library in production environments!
'''

# expose core functions
from .scoring_pipeline import score_model
from .coverage_estimator import CoverageEstimator, compute_ped, ped_null_distribution