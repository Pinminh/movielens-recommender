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

n_fold = 5
print(f"Cross validation of {n_fold} folds for Slope One on ml-latest-small dataset")
cross_validate(algo, data, measures=["rmse", "mae"], cv=n_fold, verbose=True)

print("\nAccuracy of Slope One on ml-latest-small dataset with...")
print("...75% of dataset used for training, the other 25% used for testing")
accuracy.rmse(predictions)
accuracy.mae(predictions)

print("\nNow, the model uses 100% of dataset to train")
algo.fit(full_trainset)
full_predictions = algo.test(full_testset)
store_algo("slopeone_on_ml-latest-small", algo)

uid, iid = str(1), str(6)
print(f"\nPrediction for user {uid} and movie {iid} (the true rating is 4.0)")
algo.predict(uid, iid, r_ui=4, verbose=True)

uid, iid = str(1), str(2)
print(f"\nPrediction for user {uid} and movie {iid} (no true rating exists)")
algo.predict(uid, iid, verbose=True)

n = 50
uid = str(1)
print(f"\nPrediction for top {n} movies of user {uid}")
top_movies = recommend_top_n(full_predictions, n=n)
titles_dict = load_movie_titles(dataset_name)
for movie_id, est_rating in top_movies[uid]:
    print("Rated %.1f - MovieID %s - %s" % (est_rating, movie_id, titles_dict[movie_id]))