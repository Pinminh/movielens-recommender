from surprise import SVD
from surprise.model_selection import GridSearchCV
from utils import load_movielens, seed_random_state, store_gscv


gs_name = "gs_svd_reg"
data = load_movielens()

param_grid = {
    "n_factors": [13],
    "n_epochs": [30],
    "reg_all": [0.01, 0.02, 0.05, 0.1],
}

seed_random_state()
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=10, joblib_verbose=3, n_jobs=-1)
gs.fit(data)

store_gscv(gs_name, gs)