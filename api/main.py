from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.mushroom_classification.inference import predict_frame
from src.mushroom_classification.logging_config import configure_logging
from src.mushroom_classification.monitoring import PredictionMonitor
from src.mushroom_classification.validation import FEATURE_COLUMNS

from .model_loader import ModelLoader, load_model
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    MushroomFeatures,
    PredictionResponse,
)

SAFETY_NOTICE = (
    "Academic demonstration only; never use this model to decide whether a "
    "mushroom is safe to eat."
)
logger = configure_logging()


def create_app(
    model_path: Path | None = None,
    monitor: PredictionMonitor | None = None,
) -> FastAPI:
    model_loader = ModelLoader(model_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model = model_loader.load()
        application.state.model_loader = model_loader
        logger.info("Mushroom prediction model loaded.")
        yield
        logger.info("Mushroom prediction API stopped.")

    application = FastAPI(
        title="Mushroom Classification API",
        description=SAFETY_NOTICE,
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.monitor = monitor or PredictionMonitor()

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        request_id = uuid.uuid4().hex[:12]
        request.app.state.monitor.record_failure("request_validation", request_id)
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"] if part != "body"),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        logger.warning("Rejected invalid mushroom request request_id=%s", request_id)
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Request validation failed.",
                "request_id": request_id,
                "details": details,
            },
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = getattr(request.app.state, "model", None) is not None
        return HealthResponse(status="healthy", model_loaded=loaded)

    @application.get("/ready")
    def readiness(request: Request) -> dict[str, str]:
        if getattr(request.app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="Prediction model is unavailable.")
        return {"status": "ready"}

    @application.get("/metadata", response_model=ModelMetadataResponse)
    def metadata(request: Request) -> ModelMetadataResponse:
        return ModelMetadataResponse(
            service="Mushroom Classification API",
            model_loaded=getattr(request.app.state, "model", None) is not None,
            feature_count=len(FEATURE_COLUMNS),
            class_mapping={"0": "edible", "1": "poisonous"},
            safety_notice=SAFETY_NOTICE,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: MushroomFeatures, request: Request) -> PredictionResponse:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        model = getattr(request.app.state, "model", None)
        if model is None:
            request.app.state.monitor.record_failure("model_unavailable", request_id)
            raise HTTPException(status_code=503, detail="Prediction model is unavailable.")

        try:
            features = pd.DataFrame([payload.model_dump(by_alias=True)])
            result = predict_frame(model, features).iloc[0]
            predicted_class = str(result["predicted_class"])
            prediction = 1 if predicted_class == "poisonous" else 0
            poisonous_probability = float(result["poisonous_probability"])
            latency_ms = (time.perf_counter() - started) * 1000
            request.app.state.monitor.record_prediction(
                prediction=prediction,
                poisonous_probability=poisonous_probability,
                latency_ms=latency_ms,
                request_id=request_id,
            )
            logger.info(
                "Mushroom prediction recorded request_id=%s class=%s latency_ms=%.3f",
                request_id,
                predicted_class,
                latency_ms,
            )
            return PredictionResponse(
                prediction=prediction,
                predicted_class=predicted_class,
                edible_probability=float(result["edible_probability"]),
                poisonous_probability=poisonous_probability,
                request_id=request_id,
            )
        except Exception as error:
            request.app.state.monitor.record_failure("prediction_error", request_id)
            logger.error(
                "Mushroom prediction failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=503, detail="Mushroom prediction failed."
            ) from error

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        model = getattr(request.app.state, "model", None)
        if model is None:
            request.app.state.monitor.record_failure("model_unavailable")
            raise HTTPException(status_code=503, detail="Prediction model is unavailable.")
        from .preprocessing import predict_records

        try:
            started = time.perf_counter()
            results = predict_records(model, payload.records)
            latency_per_record = (time.perf_counter() - started) * 1000 / len(payload.records)
            response_items = []
            for _, result in results.iterrows():
                predicted_class = str(result["predicted_class"])
                prediction = 1 if predicted_class == "poisonous" else 0
                request_id = uuid.uuid4().hex[:12]
                poisonous_probability = float(result["poisonous_probability"])
                request.app.state.monitor.record_prediction(
                    prediction,
                    poisonous_probability,
                    latency_per_record,
                    request_id,
                )
                response_items.append(
                    PredictionResponse(
                        prediction=prediction,
                        predicted_class=predicted_class,
                        edible_probability=float(result["edible_probability"]),
                        poisonous_probability=poisonous_probability,
                        request_id=request_id,
                    )
                )
            logger.info(
                "Mushroom batch prediction recorded count=%d latency_per_record_ms=%.3f",
                len(response_items),
                latency_per_record,
            )
            return BatchPredictionResponse(count=len(response_items), predictions=response_items)
        except Exception as error:
            request.app.state.monitor.record_failure("batch_prediction_error")
            logger.error(
                "Mushroom batch prediction failed error_type=%s",
                type(error).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=503, detail="Mushroom batch prediction failed."
            ) from error

    return application


app = create_app()
