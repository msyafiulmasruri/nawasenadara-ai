import logging

from app.schemas import ChatMessage, CounselingChatResponse, RiskLevel
from app.services.groq_client import GroqCounselingClient, should_escalate
from app.services.sentiment_model import SentimentModel

logger = logging.getLogger("nawasenadara.counseling")

_RISK_MAP: dict[str, RiskLevel] = {
    "aman": "rendah",
    "netral": "rendah",
    "sedih": "sedang",
    "takut": "tinggi",
    "marah": "sedang",
    "menyinggung": "tinggi",
}


def handle_counseling_chat(
    text: str,
    history: list[ChatMessage],
    sentiment_model: SentimentModel,
    groq_client: GroqCounselingClient,
) -> CounselingChatResponse:
    label, confidence, _scores = sentiment_model.predict(text)
    escalated = should_escalate(label, confidence)

    reply = groq_client.reply(text=text, history=history, emotion_label=label)

    return CounselingChatResponse(
        reply=reply,
        emotion_detected=label,
        emotion_confidence=round(confidence, 4),
        risk_level=_RISK_MAP.get(label, "rendah"),
        escalated=escalated,
    )
