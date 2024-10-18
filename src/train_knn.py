from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from rs_utility import load_movielens, store_algo

dataset_name = "ml-latest-small"
algo_dump_name = "knn_on_ml-latest-small"

data = load_movielens(dataset_name)
trainset = data.build_full_trainset()

sim_options = {
    "name": "pearson",
    "user_based": True,
}

algo = KNNWithZScore(min_k=5, k=100, sim_options=sim_options)
algo.fit(trainset)

store_algo(algo_dump_name, algo)