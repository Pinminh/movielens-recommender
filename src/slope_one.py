from surprise import SlopeOne, accuracy
from surprise.model_selection import cross_validate, train_test_split

from rs_utility import load_movielens, recommend_top_n, load_movie_titles, store_algo


dataset_name = "ml-latest-small"

data = load_movielens(dataset_name)
algo = SlopeOne()

# Define full trainset, testset that use all data for training/testing
full_trainset = data.build_full_trainset()
full_testset = full_trainset.build_testset()

# Define trainset, testset that use 75% for training, 25% for testing
trainset, testset = train_test_split(data, test_size=0.25)
predictions = algo.fit(trainset).test(testset)

print("Cross validation of 10 folds for Slope One on ml-latest-small dataset")
cross_validate(algo, data, measures=["rmse", "mae"], cv=10, verbose=True)

print("Accuracy of Slope One on ml-latest-small dataset with...")
print("...75% of dataset used for training, the other 25% used for testing")
accuracy.rmse(predictions)
accuracy.mae(predictions)

print("Now, the model uses 100% of dataset to train")
algo.fit(full_trainset)
full_predictions = algo.test(full_testset)
store_algo("slopeone_on_ml-latest-small", algo)

print("Prediction for user 1 and movie 6 (the true rating is 4.0)")
uid, iid = str(1), str(6)
algo.predict(uid, iid, r_ui=4, verbose=True)

print("Prediction for user 1 and movie 2 (no true rating exists)")
uid, iid = str(1), str(2)
algo.predict(uid, iid, verbose=True)

n = 50
print("Prediction for top %d movies of user 1" % n)
top_movies = recommend_top_n(full_predictions, n=n)
titles_dict = load_movie_titles(dataset_name)
uid = str(1)
for movie_id, est_rating in top_movies[uid]:
    print("Rated %.1f - MovieID %s - %s" % (est_rating, movie_id, titles_dict[movie_id]))