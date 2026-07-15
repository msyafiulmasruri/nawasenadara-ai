import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_internal_api_key
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.sentiment_model import SentimentModel, get_sentiment_model

logger = logging.getLogger("nawasenadara.router.analyze")

router = APIRouter(prefix="/analyze", tags=["NLP"])


@router.post("", response_model=AnalyzeResponse, dependencies=[Depends(verify_internal_api_key)])
def analyze_text(
    payload: AnalyzeRequest,
    model: SentimentModel = Depends(get_sentiment_model),
) -> AnalyzeResponse:
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLP belum siap.")

    try:
        label, confidence, scores = model.predict(payload.text)
    except Exception:
        logger.exception("Gagal melakukan prediksi sentimen")
        raise HTTPException(status_code=500, detail="Gagal menganalisis teks.")

    return AnalyzeResponse(label=label, confidence=confidence, scores=scores)
