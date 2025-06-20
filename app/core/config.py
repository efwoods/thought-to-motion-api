from pydantic_settings import BaseSettings

import torch


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_HOST: str  # no default
    POSTGRES_PORT: int = 5432

    # MongoDB
    MONGO_DB: str
    MONGO_HOST: str
    MONGO_PORT: int = 27017

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379

    # Ngrok / WebSocket
    FASTAPI_PORT: int = 8000
    WEBSOCKET_PORT: int = 8000

    # Torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcription
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
