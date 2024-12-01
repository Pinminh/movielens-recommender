from utils import load_movielens, load_algo, store_cv
from evals import cv_kfolds_with_measures


algo_name = "svd"
cv_name = "cv_svd"

data = load_movielens()

predictions, algo = load_algo(algo_name)

cv_result = cv_kfolds_with_measures(algo, data, n_splits=10, k=100, threshold=3.5)

store_cv(cv_name, cv_result)