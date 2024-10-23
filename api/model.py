import pandas as pd

from collections import defaultdict


__movies_df = pd.read_csv("../data/ml-latest-small/movies.csv", dtype={"movieId": str})
__links_df = pd.read_csv("../data/ml-latest-small/links.csv", dtype={"movieId": str, "imdbId": str, "tmdbId": str})

__movie_info_df = pd.merge(__movies_df, __links_df, on='movieId')
__movie_info_dict = __movie_info_df.set_index('movieId').to_dict(orient='index')


def iid_to_info(movie_id, movie_info_dict=__movie_info_dict):
    movie_id = str(movie_id)
    return movie_info_dict.get(movie_id, {})


def recommend_top_k(predictions, k=10, movie_info_dict=__movie_info_dict):
	top_k = defaultdict(list)

	for uid, iid, _, est_r, _ in predictions:
		top_k[uid].append((iid, est_r))

	for uid, movies in top_k.items():
		movies.sort(key=lambda tuple: tuple[1], reverse=True)
		top_k[uid] = movies[:k]
 
	for uid, movies in top_k.items():
		infos = []
		for iid, est_r in movies:
			info = {"movieId": iid, "est_r": est_r}
			info = info | iid_to_info(iid, movie_info_dict)
			infos.append(info)
		top_k[uid] = infos
	
	return top_k


def get_nearest_neighbors(algo, id, k=10, movie_info_dict=__movie_info_dict):
    is_user_based = algo.sim_options["user_based"]
    
    id = str(id)
    inner_id = algo.trainset.to_inner_uid(id) if is_user_based else algo.trainset.to_inner_iid(id)
    
    neighbors = algo.get_neighbors(inner_id, k)
    neighbors = [
		(algo.trainset.to_raw_uid(in_id) if is_user_based else algo.trainset.to_raw_iid(in_id)) for in_id in neighbors
	]
    
    if not is_user_based:
        neighbors = [
			{"movieId": raw_id} | iid_to_info(raw_id, movie_info_dict) for raw_id in neighbors
		]
    
    return neighbors