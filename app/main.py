from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger

# API Routes
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model

    yield  # Application runs here

    # Shutdown: (optional cleanup)
    # e.g., release resources or shutdown thread pools


app = FastAPI(
    title="Real-Time Movement Prediction Service",
    root_path="/thought-to-motion-api",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/thought-to-motion-api", tags=["Thought-To-Motion"])


@app.get("/")
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path", "") + "/docs")


@app.get("/health")
async def health():
    metrics.health_requests.inc()
    return {"status": "healthy"}


@app.router.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT)
