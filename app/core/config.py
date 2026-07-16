from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8001
    environment: str = "development"

    allowed_origins: str = "http://localhost:8000,http://localhost:3000"

    sentiment_model_dir: str = "./model"
    max_sequence_length: int = 128

    groq_api_key: str = ""
    # Default diganti dari llama-3.3-70b-versatile (di-deprecate resmi
    # oleh Groq 17 Juni 2026) ke openai/gpt-oss-120b, rekomendasi
    # migrasi resmi Groq untuk workload setara — lihat
    # https://console.groq.com/docs/deprecations
    groq_model: str = "openai/gpt-oss-120b"

    internal_api_key: str = ""

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    # lru_cache -> file .env cuma dibaca sekali per proses, bukan di
    # setiap request yang butuh Settings.
    return Settings()
