from surprise import KNNWithZScore
from utils import load_movielens, store_algo

data = load_movielens()
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

sim_options = {
    "name": "pearson",
    "user_based": True,
}

algo = KNNWithZScore(k=38, sim_options=sim_options)
predictions = algo.fit(trainset).test(testset)

algo_dump_name = "knn_zscore_full"
store_algo(algo_dump_name, predictions, algo)