import sys, os
sys.path.append(os.path.abspath("../"))

from fastapi import FastAPI, HTTPException
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from model import recommend_top_k, get_nearest_neighbors
from core.utils import load_algo

app = FastAPI()

slope_pred, slope_algo = load_algo("slope_one_full")
knn_pred, knn_algo = load_algo("knn_zscore_full")

models = {
    "slope_one": (slope_pred, slope_algo),
    "knn_zscore": (knn_pred, knn_algo),
}


@app.get("/predict")
async def predict(uid: int, iid: int, model: str = "slope_one"):
    if model not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    
    _, algo = models[model]
    pred = algo.predict(uid=str(uid), iid=str(iid), clip=False)
    
    return {"uid": uid, "iid": iid, "est_r": pred.est, "details": pred.details}
    

@app.get("/recommend")
async def recommend(uid: int, k: int = 10, model: str = "slope_one"):
    if model not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    pred, _ = models[model]
    top_k = recommend_top_k(pred, k)

    return {"uid": uid, "top_k": top_k[str(uid)]}
    


@app.get("/neighbors")
async def neighbors(uid: int, k: int = 10, model: str = "knn_zscore"):
    if model not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    
    _, algo = models[model]
    if not isinstance(algo, (KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline)):
        raise HTTPException(status_code=422, detail=f"Model '{model}' no similarity measure")
    
    neighbors = get_nearest_neighbors(algo, str(uid), k)
    
    return {"uid": uid, "neighbors": neighbors}
