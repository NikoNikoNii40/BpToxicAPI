import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# =========================
#  Configuration (Step 1)
# =========================

LABELS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "insult",
    "threat",
    "identity_hate",
]

MODEL_ID = os.environ.get("MODEL_ID", "xlmr-base-v1")
THRESHOLD_SET = os.environ.get("THRESHOLD_SET", "per_label_v1")

# Input limits (you can tune later)
MAX_CHARS = int(os.environ.get("MAX_CHARS", "4000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

# Strictness slider behavior:
# thresholds_s = clamp(base + (strictness - 0.5) * DELTA, MIN_T, MAX_T)
STRICTNESS_DELTA = float(os.environ.get("STRICTNESS_DELTA", "0.25"))
MIN_T = float(os.environ.get("MIN_THRESHOLD", "0.05"))
MAX_T = float(os.environ.get("MAX_THRESHOLD", "0.95"))

# Optional API auth
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "").strip()

# HF model path or name (later point this to your fine-tuned checkpoint folder)
HF_MODEL_PATH = os.environ.get("HF_MODEL_PATH", "xlm-roberta-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "thresholds.json")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")

app = FastAPI(title="Toxicity Remote Model API", version="1.0.0")

# =========================
#  In-memory latency stats (privacy-safe)
# =========================
LAT_COUNT = 0
LAT_TOTAL_SUM = 0
LAT_MODEL_SUM = 0
LAT_LAST_200_TOTAL: List[int] = []
LAT_LAST_200_MODEL: List[int] = []

# =========================
#  Request / Response Models
# =========================

class PredictRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Single input text")
    texts: Optional[List[str]] = Field(default=None, description="Batch input texts")
    lang: Optional[str] = Field(default=None, description="Optional language code, e.g. 'en'")
    strictness: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="0=lenient, 1=strict")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("text must be a string")
        return v

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if v is None:
            return v
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            raise ValueError("texts must be a list of strings")
        return v

    @field_validator("strictness")
    @classmethod
    def validate_strictness(cls, v):
        if v is None:
            return 0.5
        return float(v)


class Verdict(BaseModel):
    is_toxic: bool
    triggered_labels: List[str]


class PredictResponse(BaseModel):
    scores: Dict[str, float]
    verdict: Verdict
    meta: Dict[str, object]

# =========================
#  Utility: security
# =========================

def require_bearer_token(request: Request) -> None:
    if not API_BEARER_TOKEN:
        return  # auth disabled
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# =========================
#  Utility: preprocessing
# =========================

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub("<URL>", s)
    s = USER_RE.sub("<USER>", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def enforce_char_limit(s: str) -> str:
    if len(s) > MAX_CHARS:
        s = s[:MAX_CHARS]
    return s

# =========================
#  Thresholds & strictness
# =========================

def load_base_thresholds() -> Dict[str, float]:
    if not os.path.exists(THRESHOLDS_PATH):
        raise RuntimeError(f"Missing thresholds file: {THRESHOLDS_PATH}")
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in LABELS:
        if k not in data:
            raise RuntimeError(f"Threshold '{k}' missing in {THRESHOLDS_PATH}")
    return {k: float(data[k]) for k in LABELS}


BASE_THRESHOLDS = load_base_thresholds()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def thresholds_for_strictness(strictness: float) -> Dict[str, float]:
    shift = (strictness - 0.5) * STRICTNESS_DELTA
    return {k: clamp(BASE_THRESHOLDS[k] + shift, MIN_T, MAX_T) for k in LABELS}


def compute_verdict(scores: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
    triggered = [k for k in LABELS if float(scores.get(k, 0.0)) >= float(thresholds[k])]
    return (len(triggered) > 0), triggered

# =========================
#  Model loading & inference
# =========================

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None


def load_model() -> None:
    global _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, use_fast=True)

    # Force 6-label multi-label head
    config = AutoConfig.from_pretrained(HF_MODEL_PATH)
    config.num_labels = len(LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = {i: LABELS[i] for i in range(len(LABELS))}
    config.label2id = {LABELS[i]: i for i in range(len(LABELS))}

    _model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_PATH,
        config=config,
        ignore_mismatched_sizes=True,
    )

    _model.to(DEVICE)
    _model.eval()

    print(f"[OK] Loaded model '{HF_MODEL_PATH}' with num_labels={_model.config.num_labels} on {DEVICE}")


@torch.inference_mode()
def predict_scores(texts: List[str]) -> List[Dict[str, float]]:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded yet")

    enc = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = _model(**enc).logits  # (batch, num_labels)
    probs = torch.sigmoid(logits).detach().cpu().tolist()

    results: List[Dict[str, float]] = []
    for row in probs:
        row = list(row)
        if len(row) < len(LABELS):
            row += [0.0] * (len(LABELS) - len(row))
        row = row[: len(LABELS)]
        results.append({LABELS[i]: float(row[i]) for i in range(len(LABELS))})
    return results


def warmup_model() -> None:
    """
    Warm up CUDA kernels / allocate memory by running a couple of dummy inferences.
    This reduces cold-start latency for the first real request.
    """
    try:
        _ = predict_scores(["warmup", "hello world"])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        print("[warmup] done")
    except Exception as e:
        print(f"[warmup] failed: {e}")


@app.on_event("startup")
def _startup() -> None:
    load_model()
    warmup_model()

# =========================
#  Endpoints
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_id": MODEL_ID,
        "hf_model_path": HF_MODEL_PATH,
        "max_chars": MAX_CHARS,
        "max_tokens": MAX_TOKENS,
        "labels": LABELS,
    }


@app.get("/stats")
def stats():
    """
    Privacy-safe latency stats (stores only numbers, no text).
    p95 is estimated from the last 200 requests.
    """
    if LAT_COUNT == 0:
        return {"count": 0}

    def p95(values):
        if not values:
            return 0
        vals = sorted(values)
        # "nearest-rank" method:
        # rank = ceil(0.95 * N) -> 1..N
        import math
        rank = max(1, math.ceil(0.95 * len(vals)))
        return vals[rank - 1]

    return {
        "count": LAT_COUNT,
        "avg_total_ms": round(LAT_TOTAL_SUM / LAT_COUNT, 2),
        "avg_model_ms": round(LAT_MODEL_SUM / LAT_COUNT, 2),
        "p95_total_ms_last200": p95(LAT_LAST_200_TOTAL),
        "p95_model_ms_last200": p95(LAT_LAST_200_MODEL),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    global LAT_COUNT, LAT_TOTAL_SUM, LAT_MODEL_SUM, LAT_LAST_200_TOTAL, LAT_LAST_200_MODEL

    require_bearer_token(request)

    if req.text is None and (req.texts is None or len(req.texts) == 0):
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

    strictness = float(req.strictness or 0.5)
    thr = thresholds_for_strictness(strictness)

    texts_in = [req.text] if req.text is not None else (req.texts or [])

    processed: List[str] = []
    for t in texts_in:
        t2 = normalize_text(enforce_char_limit(t))
        processed.append(t2)

    # ---- latency measurement ----
    t_total_start = time.perf_counter()

    t_model_start = time.perf_counter()
    scores_list = predict_scores(processed)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    model_latency_ms = int((time.perf_counter() - t_model_start) * 1000)

    total_latency_ms = int((time.perf_counter() - t_total_start) * 1000)

    # Update rolling stats (numbers only)
    LAT_COUNT += 1
    LAT_TOTAL_SUM += total_latency_ms
    LAT_MODEL_SUM += model_latency_ms
    LAT_LAST_200_TOTAL.append(total_latency_ms)
    LAT_LAST_200_MODEL.append(model_latency_ms)
    LAT_LAST_200_TOTAL = LAT_LAST_200_TOTAL[-200:]
    LAT_LAST_200_MODEL = LAT_LAST_200_MODEL[-200:]

    scores = scores_list[0]
    is_toxic, triggered = compute_verdict(scores, thr)

    return PredictResponse(
        scores=scores,
        verdict=Verdict(is_toxic=is_toxic, triggered_labels=triggered),
        meta={
            "mode": "remote",
            "model_id": MODEL_ID,
            "threshold_set": THRESHOLD_SET,
            "model_latency_ms": model_latency_ms,
            "total_latency_ms": total_latency_ms,
            "lang": req.lang,
            "strictness": strictness,
            "max_chars": MAX_CHARS,
            "max_tokens": MAX_TOKENS,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})