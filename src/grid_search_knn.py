from surprise import KNNWithZScore
from surprise.model_selection import GridSearchCV
from rs_utility import load_movielens


param_grid = {
    "k": [10, 40],
    "sim_options": {
        "name": ["msd", "cosine", "pearson"],
        "user_based": [False, True],
    }
}

data = load_movielens("ml-latest-small")
gs = GridSearchCV(KNNWithZScore, param_grid, measures=["mae", "rmse"], cv=10)
gs.fit(data)

print(gs.best_score["mae"])
print(gs.best_params["mae"])

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
