from surprise import SVD
from utils import load_movielens, store_algo


data = load_movielens()
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algo = SVD(n_factors=13, n_epochs=30, reg_all=0.04, verbose=True)
predictions = algo.fit(trainset).test(testset)

algo_dump_name = "svd"
store_algo(algo_dump_name, predictions, algo)