import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import analyze, chat
from app.schemas import HealthResponse
from app.services.groq_client import get_groq_client
from app.services.sentiment_model import get_sentiment_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nawasenadara.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model dimuat SEKALI saat proses start-up (bukan per-request) —
    # loading + konversi bobot PyTorch->TensorFlow makan waktu
    # beberapa detik, jadi harus terjadi sekali saja di awal.
    logger.info("Starting up Nawasena Dara NLP Service...")
    model = get_sentiment_model()
    try:
        model.load()
    except Exception:
        # Sengaja TIDAK menghentikan startup kalau model gagal dimuat
        # (mis. karena file bobot belum ditaruh) — /health akan
        # melaporkan model_loaded: false, dan endpoint /analyze &
        # /chat/counseling akan mengembalikan 503 sampai model tersedia.
        # Ini supaya service tetap bisa dites (mis. endpoint /health)
        # bahkan sebelum file bobot model ditambahkan.
        logger.exception(
            "Model NLP gagal dimuat — pastikan file bobot model "
            "(pytorch_model.bin / model.safetensors) sudah ditaruh di "
            "SENTIMENT_MODEL_DIR."
        )

    yield
    logger.info("Shutting down Nawasena Dara NLP Service...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Nawasena Dara — NLP & Chatbot Konseling Service",
        description=(
            "Microservice FastAPI: analisis sentimen/emosi (IndoBERT via "
            "TensorFlow) + chatbot konseling virtual (Groq/Llama). "
            "Dikonsumsi secara internal oleh nawasenadara-backend (Express)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(analyze.router)
    app.include_router(chat.router)

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health() -> HealthResponse:
        model = get_sentiment_model()
        groq_client = get_groq_client()
        return HealthResponse(
            status="ok",
            model_loaded=model.is_loaded,
            groq_configured=groq_client.is_configured,
        )

    @app.get("/", tags=["System"])
    def root():
        return {
            "service": "Nawasena Dara NLP & Chatbot Konseling",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
