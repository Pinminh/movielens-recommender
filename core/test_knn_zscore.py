from utils import load_movielens, load_algo, store_cv
from utils import RANK_K, RANK_THRESHOLD, CV_FOLDS
from evals import cv_kfolds_with_measures


algo_name = "knn_zscore_full"
cv_name = "cv_knn_zscore"

data = load_movielens()

predictions, algo = load_algo(algo_name)

cv_result = cv_kfolds_with_measures(algo, data, n_splits=CV_FOLDS, k=RANK_K, threshold=RANK_THRESHOLD)

store_cv(cv_name, cv_result)