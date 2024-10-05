import os
from collections import defaultdict

from surprise import Dataset, Reader, dump



__movielens_reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

__dataset_path = os.path.relpath("../data/")
__algo_dump_path = os.path.relpath("../output/algorithms/")

__ratings_fname = "ratings.csv"
__movies_fname = "movies.csv"



def load_movielens(dataset_name):
    file_path = os.path.join(__dataset_path, dataset_name, __ratings_fname)
    return Dataset.load_from_file(file_path, reader=__movielens_reader)



def store_algo(algo_name, algo):
    file_path = os.path.join(__algo_dump_path, algo_name)
    dump.dump(file_path, algo=algo)

def load_algo(algo_name):
    file_path = os.path.join(__algo_dump_path, algo_name)
    return dump.load(file_path)



def load_movie_titles(dataset_name):
    movies_path = os.path.join(__dataset_path, dataset_name, __movies_fname)
    movie_dict = defaultdict(list)
    
    with open(movies_path, encoding="ISO-8859-1") as file:
        for line in file:
            fields = line.split(",")
            movie_id = fields[0]
            movie_title = fields[1]
            movie_dict[movie_id] = movie_title
            
    return movie_dict



def recommend_top_n(predictions, n=10):
	top_n = defaultdict(list)

	for user, item, _, estimation, _ in predictions:
		top_n[user].append((item, estimation))

	for user, ratings in top_n.items():
		ratings.sort(key=lambda tuple: tuple[1], reverse=True)
		top_n[user] = ratings[:n]

	return top_n



if __name__ == "__main__":
    print("rs_utility module contains shortcuts for dumping, loading algorithms, and recommending functions...")