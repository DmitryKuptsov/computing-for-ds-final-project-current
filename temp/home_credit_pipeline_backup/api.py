import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from home_credit.feature_engineer import FeatureEngineer

MODEL_PATH = "artifacts/lgbm_model.joblib"
TRANSFORMER_PATH = "artifacts/preprocessor.joblib"
DATA_PATH = "data/application_train.csv"

model = None
transformer = None
raw_df = None
fe = FeatureEngineer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transformer, raw_df
    # startup
    model = joblib.load(MODEL_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)
    raw_df = pd.read_csv(DATA_PATH)
    yield
    # shutdown (optional cleanup)
    # e.g. close DB connections


app = FastAPI(
    title="Home Credit Default Risk API",
    lifespan=lifespan,
)


class PredictByIdRequest(BaseModel):
    sk_id_curr: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "transformer_loaded": transformer is not None,
        "data_path": DATA_PATH,
        "rows_loaded": 0 if raw_df is None else int(len(raw_df)),
    }


@app.post("/predict_by_id")
def predict_by_id(req: PredictByIdRequest):
    if raw_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    if "SK_ID_CURR" not in raw_df.columns:
        raise HTTPException(
            status_code=500,
            detail="SK_ID_CURR column not found in loaded data",
        )

    row = raw_df.loc[raw_df["SK_ID_CURR"] == req.sk_id_curr]
    if row.empty:
        raise HTTPException(status_code=404, detail="SK_ID_CURR not found")

    X = row.copy()
    if "TARGET" in X.columns:
        X = X.drop(columns=["TARGET"])

    # must match training: FeatureEngineer -> transformer -> model
    X = fe.transform(X)
    Xt = transformer.transform(X)
    proba = float(model.predict_proba(Xt)[:, 1][0])

    return {
        "sk_id_curr": req.sk_id_curr,
        "default_probability": proba,
    }
