import numpy as np

from surprise import SVDpp
from surprise.model_selection import GridSearchCV
from utils import load_movielens, seed_random_state, store_gscv


gs_name = "gs_svdpp"
data = load_movielens()

param_grid = {
    "n_factors": [10, 15, 20, 25, 30],
    "n_epochs": [30],
}

seed_random_state()
gs = GridSearchCV(SVDpp, param_grid, measures=["rmse", "mae"], cv=10, joblib_verbose=3, n_jobs=-1)
gs.fit(data)

store_gscv(gs_name, gs)