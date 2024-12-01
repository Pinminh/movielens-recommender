from surprise import SVDpp
from utils import load_movielens, store_algo


data = load_movielens()
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algo = SVDpp(n_factors=11, n_epochs=30, reg_all=0.06, verbose=True)
algo.fit(trainset)

algo_dump_name = "svdpp"
store_algo(algo_dump_name, algo=algo)