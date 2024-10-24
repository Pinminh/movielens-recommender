import os
import random
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from surprise import Dataset, Reader, dump


__movielens_reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

CURRENT_PATH = Path(__file__)
PROJECT_PATH = CURRENT_PATH.parent.parent
DATASETS_PATH = PROJECT_PATH.joinpath("data")
ALGO_DUMP_PATH = PROJECT_PATH.joinpath("output", "models")
CV_DUMP_PATH = PROJECT_PATH.joinpath("output", "cv")
GSCV_DUMP_PATH = PROJECT_PATH.joinpath("output", "gscv")

DEFAULT_DATASET_NAME = "ml-latest-small"
RATINGS_FILENAME = "ratings.csv"


def seed_random_state(seed=250704):
    random.seed(seed)
    np.random.seed(seed)


def trainset_to_dataset(trainset):
    df = pd.DataFrame(trainset.build_testset(), columns=["userId", "movieId", "rating"])
    return Dataset.load_from_df(df[["userId", "movieId", "rating"]], __movielens_reader)


def load_movielens(dataset_name=DEFAULT_DATASET_NAME):
    file_path = DATASETS_PATH.joinpath(dataset_name, RATINGS_FILENAME)
    return Dataset.load_from_file(file_path, reader=__movielens_reader)


def store_algo(algo_name: str, pred=None, algo=None):
    file_path = ALGO_DUMP_PATH.joinpath(algo_name)
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    dump.dump(file_path, pred, algo)


def load_algo(algo_name: str):
    file_path = ALGO_DUMP_PATH.joinpath(algo_name)
    return dump.load(file_path)


def store_cv(cv_name: str, cv_result):
    file_path = CV_DUMP_PATH.joinpath(cv_name)
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, "wb") as file:
        pickle.dump(cv_result, file, pickle.HIGHEST_PROTOCOL)


def load_cv(cv_name: str):
    file_path = CV_DUMP_PATH.joinpath(cv_name)
    
    with open(file_path, "rb") as file:
        cv_result = pickle.load(file)
    
    return cv_result


def store_gscv(gscv_name: str, gscv):
    file_path = GSCV_DUMP_PATH.joinpath(gscv_name)
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, "wb") as file:
        pickle.dump(gscv, file, pickle.HIGHEST_PROTOCOL)


def load_gscv(gscv_name: str):
    file_path = GSCV_DUMP_PATH.joinpath(gscv_name)
    
    with open(file_path, "rb") as file:
        gscv = pickle.load(file)
    
    return gscv


if __name__ == "__main__":
    file_name = os.path.basename(__file__)
    module_name = os.path.splitext(file_name)[0]
    
    print(f"{module_name} module contains functions for dumping and loading datasets, algorithms, gridsearch results...")