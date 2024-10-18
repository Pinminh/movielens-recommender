from rs_utility import load_movielens, load_algo, recommend_top_n, load_movie_titles


dataset_name = "ml-latest-small"
algo_dump_name = "knn_on_ml-latest-small"

data = load_movielens(dataset_name)
testset = data.build_full_trainset().build_testset()

_, algo = load_algo(algo_dump_name)

full_predictions = algo.test(testset)

print("\nPrediction for user 1 and movie 6 (the true rating is 4.0)")
uid, iid = str(1), str(6)
algo.predict(uid, iid, r_ui=4, verbose=True)

print("\nPrediction for user 1 and movie 2 (no true rating exists)")
uid, iid = str(1), str(2)
algo.predict(uid, iid, verbose=True)

n = 50
print("\nPrediction for top %d movies of user 1" % n)
top_movies = recommend_top_n(full_predictions, n=n)
titles_dict = load_movie_titles(dataset_name)
uid = str(1)
for movie_id, est_rating in top_movies[uid]:
    print("Rated %.1f - MovieID %s - %s" % (est_rating, movie_id, titles_dict[movie_id]))