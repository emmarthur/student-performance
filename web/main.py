"""
FastAPI app — student pass/fail prediction (decision tree).

Run: uvicorn main:app --reload --port 3028
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from student_performance_core import (  # noqa: E402
    INPUT_FIELDS,
    default_student_fields,
    field_options,
    get_or_train_model,
    load_raw_frame,
)

_model = None
_training = False
_train_error: str | None = None
_lock = threading.Lock()


def _train_in_background() -> None:
    global _model, _training, _train_error
    with _lock:
        if _model is not None or _training:
            return
        _training = True
        _train_error = None
    try:
        trained = get_or_train_model()
        with _lock:
            _model = trained
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _train_error = str(exc)
    finally:
        with _lock:
            _training = False


def _ensure_training_started() -> None:
    threading.Thread(target=_train_in_background, daemon=True).start()


def _status_payload() -> dict[str, Any]:
    with _lock:
        if _model is None:
            return {
                "ready": False,
                "training": _training,
                "error": _train_error,
            }
        return {
            "ready": True,
            "training": False,
            "error": None,
            "holdout_accuracy": _model.holdout_accuracy,
            "cv_mean": _model.cv_mean,
            "cv_std": _model.cv_std,
            "pass_rate": _model.pass_rate,
            "samples": _model.sample_students,
        }


app = FastAPI(title="Student Performance Predictor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.on_event("startup")
async def startup() -> None:
    _ensure_training_started()


class PredictRequest(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "help.html")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    if _model is None and not _training and _train_error is None:
        _ensure_training_started()
    return JSONResponse(_status_payload())


@app.get("/api/schema")
async def api_schema() -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    options = field_options(load_raw_frame())
    defaults = default_student_fields(options)
    return JSONResponse(
        {
            "fields": INPUT_FIELDS,
            "options": options,
            "defaults": defaults,
        }
    )


@app.post("/api/predict")
async def api_predict(body: PredictRequest) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    try:
        result = _model.predict(body.fields)
        return JSONResponse(result)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
