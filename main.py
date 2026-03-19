from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings
import os

app = FastAPI(title="Risk Engine", version="3.0-percentile")

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_anomaly_score.pkl")
API_SECRET = os.getenv("RISK_API_SECRET", "change-me")


DEFAULT_LOW_PCT = float(os.getenv("RISK_LOW_PERCENTILE", "5"))
DEFAULT_HIGH_PCT = float(os.getenv("RISK_HIGH_PERCENTILE", "95"))

EPS = 1e-12

try:
    loaded = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

# Support both bundled dict and plain sklearn model.
if isinstance(loaded, dict):
    bundle = loaded
    clf = bundle["model"] if "model" in bundle else loaded
else:
    bundle = {}
    clf = loaded

FEATURE_COLS = bundle.get("feature_cols") if isinstance(bundle, dict) else None
TRAIN_SCORE_MIN = float(bundle.get("train_score_min", -0.6667140844647449))
TRAIN_SCORE_MAX = float(bundle.get("train_score_max", -0.40973622752350913))
TRAIN_SCORE_RANGE = float(bundle.get("train_score_range", TRAIN_SCORE_MAX - TRAIN_SCORE_MIN))

# Read real percentiles if present in model bundle.
BUNDLE_P05 = bundle.get("train_score_p05")
BUNDLE_P50 = bundle.get("train_score_p50")
BUNDLE_P95 = bundle.get("train_score_p95")


def _inner_anchor_from_minmax(min_score: float, max_score: float, pct: float) -> float:
    pct = max(0.0, min(100.0, pct)) / 100.0
    return float(min_score + (max_score - min_score) * pct)


def resolve_percentile_anchors() -> tuple[float, float, str]:
    if BUNDLE_P05 is not None and BUNDLE_P95 is not None:
        low_anchor = float(BUNDLE_P05)
        high_anchor = float(BUNDLE_P95)
        mode = "true_percentile_from_bundle"
    else:
        low_anchor = _inner_anchor_from_minmax(TRAIN_SCORE_MIN, TRAIN_SCORE_MAX, DEFAULT_LOW_PCT)
        high_anchor = _inner_anchor_from_minmax(TRAIN_SCORE_MIN, TRAIN_SCORE_MAX, DEFAULT_HIGH_PCT)
        mode = "trimmed_range_fallback"

    # Safety correction if anchors are reversed or collapsed.
    if high_anchor <= low_anchor:
        low_anchor = TRAIN_SCORE_MIN + 0.10 * TRAIN_SCORE_RANGE
        high_anchor = TRAIN_SCORE_MAX - 0.10 * TRAIN_SCORE_RANGE
        mode = f"{mode}_auto_corrected"

    return float(low_anchor), float(high_anchor), mode


LOW_ANCHOR, HIGH_ANCHOR, NORMALIZATION_MODE = resolve_percentile_anchors()


class Stats(BaseModel):
    m: float = Field(ge=0, le=1)
    s: float = Field(ge=0, le=1)


class Features(BaseModel):
    density: float = Field(ge=0, le=1)
    idle_ratio: float = Field(ge=0, le=1)


class BehaviorPayload(BaseModel):
    mouse: Stats
    click: Stats
    key: Stats
    idle: Stats
    features: Features


def to_vector(b: BehaviorPayload) -> list[float]:
    vec = [
        b.mouse.m, b.mouse.s,
        b.click.m, b.click.s,
        b.key.m, b.key.s,
        b.idle.m, b.idle.s,
        b.features.density,
        b.features.idle_ratio,
    ]

    if FEATURE_COLS is not None and len(FEATURE_COLS) != len(vec):
        raise HTTPException(status_code=500, detail="Feature dimension mismatch with trained model")

    return vec


def normalize(raw_score: float) -> float:
    denom = HIGH_ANCHOR - LOW_ANCHOR
    if denom <= EPS:
        return 0.0

    risk = (HIGH_ANCHOR - raw_score) / denom
    return float(np.clip(risk, 0.0, 1.0))


@app.post("/score")
def score(
    payload: BehaviorPayload,
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vec = to_vector(payload)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = float(clf.score_samples([vec])[0])
        decision = float(clf.decision_function([vec])[0])

    return {
        "raw_score": round(raw, 6),
        "decision": round(decision, 6),
        "normalized": round(normalize(raw), 6),
        "normalization_mode": NORMALIZATION_MODE,
        "low_anchor": round(LOW_ANCHOR, 6),
        "high_anchor": round(HIGH_ANCHOR, 6),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "has_bundle": isinstance(loaded, dict),
        "has_feature_cols": FEATURE_COLS is not None,
        "train_score_min": round(TRAIN_SCORE_MIN, 6),
        "train_score_max": round(TRAIN_SCORE_MAX, 6),
        "train_score_range": round(TRAIN_SCORE_RANGE, 6),
        "bundle_train_score_p05": None if BUNDLE_P05 is None else round(float(BUNDLE_P05), 6),
        "bundle_train_score_p50": None if BUNDLE_P50 is None else round(float(BUNDLE_P50), 6),
        "bundle_train_score_p95": None if BUNDLE_P95 is None else round(float(BUNDLE_P95), 6),
        "normalization_mode": NORMALIZATION_MODE,
        "low_anchor": round(LOW_ANCHOR, 6),
        "high_anchor": round(HIGH_ANCHOR, 6),
        "low_percentile": DEFAULT_LOW_PCT,
        "high_percentile": DEFAULT_HIGH_PCT,
    }