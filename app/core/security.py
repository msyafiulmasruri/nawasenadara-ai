from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verify_internal_api_key(
    x_internal_api_key: str = Header(default=""),
) -> None:
    settings = get_settings()

    # Kalau INTERNAL_API_KEY belum diisi di .env (mis. saat masih
    # development lokal), pengecekan dilewati — supaya tidak menghambat
    # development awal. WAJIB diisi sebelum deploy ke production.
    if not settings.internal_api_key:
        return

    if x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Internal-Api-Key tidak valid.",
        )
