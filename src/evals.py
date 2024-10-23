import os

from collections import defaultdict
from surprise.model_selection import KFold, cross_validate

from utils import seed_random_state


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    f1_scores = dict()
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        f1_scores[uid] = 2 * (precisions[uid] * recalls[uid]) / (precisions[uid] + recalls[uid]) if precisions[uid] + recalls[uid] != 0 else 0
    
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
    avg_f1_score = sum(f1 for f1 in f1_scores.values()) / len(f1_scores) # macro F1-score computation
        
    return precisions, recalls, f1_scores, avg_precision, avg_recall, avg_f1_score


def cv_kfolds_with_measures(algo, data, n_splits: int = 10, k: int = 10, threshold=3.5, n_jobs=-1):
    kf = KFold(n_splits=n_splits)

    seed_random_state()
    cv_result = cross_validate(algo, data, measures=["rmse", "mae"], cv=kf, n_jobs=n_jobs, verbose=False)
    
    seed_random_state()
    cv_result.update(test_prec=[], test_rec=[], test_f1=[])
    for trainset, testset in kf.split(data):
        pred = algo.fit(trainset).test(testset)
        _, _, _, precision, recall, f1_score = precision_recall_at_k(pred, k, threshold)
        cv_result["test_prec"].append(precision)
        cv_result["test_rec"].append(recall)
        cv_result["test_f1"].append(f1_score)
        
    return cv_result


def eval_precision_recall_range(predictions, k_range, threshold=3.5, verbose=False):
    dict = defaultdict(list)
    
    for k in k_range:
        print(f"Processing k={k}") if verbose else None
        _, _, _, prec, rec, f1 = precision_recall_at_k(predictions, k, threshold)
        dict[k] = {"precision": prec, "recall": rec, "f1_score": f1}
    
    max_precision_k = max(dict, key=lambda k: dict[k]["precision"])
    max_recall_k = max(dict, key=lambda k: dict[k]["recall"])
    max_f1_k = max(dict, key=lambda k: dict[k]["f1_score"])
    
    return dict, max_precision_k, max_recall_k, max_f1_k


if __name__ == "__main__":
    file_name = os.path.basename(__file__)
    module_name = os.path.splitext(file_name)[0]
    
    print(f"{module_name} module contains functions for computing additional metrics, and performing cross-validations.")