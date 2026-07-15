import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_internal_api_key
from app.schemas import CounselingChatRequest, CounselingChatResponse
from app.services.counseling import handle_counseling_chat
from app.services.groq_client import GroqCounselingClient, get_groq_client
from app.services.sentiment_model import SentimentModel, get_sentiment_model

logger = logging.getLogger("nawasenadara.router.chat")

router = APIRouter(prefix="/chat", tags=["Chatbot Konseling"])


@router.post(
    "/counseling",
    response_model=CounselingChatResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
def counseling_chat(
    payload: CounselingChatRequest,
    model: SentimentModel = Depends(get_sentiment_model),
    groq_client: GroqCounselingClient = Depends(get_groq_client),
) -> CounselingChatResponse:
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLP belum siap.")
    if not groq_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY belum dikonfigurasi di server NLP.",
        )

    try:
        return handle_counseling_chat(
            text=payload.text,
            history=payload.history,
            sentiment_model=model,
            groq_client=groq_client,
        )
    except Exception:
        logger.exception("Gagal memproses chat konseling")
        raise HTTPException(
            status_code=500, detail="Gagal memproses permintaan chatbot."
        )
