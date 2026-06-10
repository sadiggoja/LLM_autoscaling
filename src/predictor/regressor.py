from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import warnings
import logging

warnings.filterwarnings(action='ignore', category=UserWarning)
logging.basicConfig(level=logging.CRITICAL)

# CPU-only heterogeneous workload. Same /predict API shape as the localization
# regressor (POST {"feature": int} -> {"x": ..., "y": ...}) so the benchmark
# runner, load generator, and ingress configs work unchanged. Per-request CPU
# is dominated by walking N_ESTIMATORS boosted trees; memory footprint stays
# small and fixed (no pickle to ship).
N_FEATURES = 20
N_SAMPLES = 2000
N_ESTIMATORS = 500
MAX_DEPTH = 8
POOL_SIZE = 130  # matches the 0..129 feature index used by spam_cluster.py

_rng = np.random.default_rng(0)
_X_train = _rng.random((N_SAMPLES, N_FEATURES))
_y_train_x = _rng.random(N_SAMPLES)
_y_train_y = _rng.random(N_SAMPLES)
model_x = GradientBoostingRegressor(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=0
).fit(_X_train, _y_train_x)
model_y = GradientBoostingRegressor(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=1
).fit(_X_train, _y_train_y)

feature_pool = _rng.random((POOL_SIZE, N_FEATURES))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    feature: int

@app.post("/predict")
async def predict(item: Item):
    try:
        x_vec = feature_pool[item.feature % POOL_SIZE].reshape(1, -1)
        return {"x": float(model_x.predict(x_vec)[0]),
                "y": float(model_y.predict(x_vec)[0])}
    except Exception:
        raise HTTPException(status_code=400, detail="Error making prediction")
