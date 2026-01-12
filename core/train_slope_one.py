from surprise import SlopeOne
from utils import load_movielens, store_algo


data = load_movielens()
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algo = SlopeOne()
predictions = algo.fit(trainset).test(testset)

algo_dump_name = "slope_one_full"
store_algo(algo_dump_name, predictions, algo)