from typing import Literal
from pydantic import BaseModel, Field

# 6 label sentimen/emosi persis sesuai config.json model IndoBERT yang
# sudah dilatih (lihat id2label): aman, menyinggung, takut, marah,
# netral, sedih.
EmotionLabel = Literal["aman", "menyinggung", "takut", "marah", "netral", "sedih"]

# Heuristik tingkat risiko v1 — dipetakan langsung dari label emosi.
# CATATAN: ini BUKAN model deteksi darurat/intent yang sesungguhnya
# (itu dataset & model terpisah, lihat kolom "AI Engineer NLP" di
# checklist tech stack: intent curhat/bertanya/minta_saran/
# minta_bantuan/darurat, masih tahap "to do"). Selama model intent itu
# belum ada, endpoint /chat/counseling di sini memakai heuristik
# sederhana ini sebagai sinyal awal untuk memutuskan nada respons &
# kapan menyisipkan ajakan halus ke bantuan profesional.
RiskLevel = Literal["rendah", "sedang", "tinggi"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class AnalyzeResponse(BaseModel):
    label: EmotionLabel
    confidence: float
    scores: dict[EmotionLabel, float]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CounselingChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    # Riwayat percakapan sebelumnya (opsional) — dikirim balik oleh
    # backend Express setiap kali memanggil endpoint ini, supaya Groq
    # punya konteks percakapan multi-turn (service ini sendiri tidak
    # menyimpan riwayat percakapan di memori/DB, itu tanggung jawab
    # backend Express + PostgreSQL sesuai proposal 3.1.2).
    history: list[ChatMessage] = Field(default_factory=list)
    user_name: str | None = None


class CounselingChatResponse(BaseModel):
    reply: str
    emotion_detected: EmotionLabel
    emotion_confidence: float
    risk_level: RiskLevel
    # True kalau respons chatbot sudah menyisipkan ajakan mencari
    # bantuan profesional/orang dewasa terpercaya — dipakai backend
    # Express untuk memutuskan apakah perlu memicu notifikasi darurat
    # ke dashboard guru BK (lihat proposal Bab 2.2.6 & Gambar 3.1).
    escalated: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    groq_configured: bool
