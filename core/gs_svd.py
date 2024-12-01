import numpy as np

from surprise import SVD
from surprise.model_selection import GridSearchCV
from utils import load_movielens, seed_random_state, store_gscv


gs_name = "gs_svd"
data = load_movielens()

param_grid = {
    "n_factors": np.arange(10, 101, 10),
    "n_epochs": np.arange(10, 31, 10),
}

seed_random_state()
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=10, joblib_verbose=3, n_jobs=2)
gs.fit(data)

store_gscv(gs_name, gs)