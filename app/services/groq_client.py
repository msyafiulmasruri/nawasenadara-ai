"""
Integrasi Groq API (Llama) untuk persona Chatbot Konseling Virtual —
sesuai proposal Bab 2.2.6: empatik, suportif, tidak menghakimi, dan
TIDAK dimaksudkan menggantikan layanan konseling profesional.
"""

import logging

from groq import Groq

from app.core.config import get_settings
from app.schemas import ChatMessage, EmotionLabel

logger = logging.getLogger("nawasenadara.groq_client")

# Prompt sistem persona konselor virtual. Konteks emosi (hasil model
# IndoBERT) disisipkan sebagai instruksi tambahan di setiap panggilan —
# BUKAN dikirim sebagai fakta ke pemain, supaya bot tetap terasa
# menyimak secara alami, bukan seperti membacakan hasil analisis.
SYSTEM_PROMPT_TEMPLATE = """Kamu adalah "Kak Dara", konselor virtual di aplikasi Nawasena Dara — \
sebuah learning game edukasi pencegahan kekerasan untuk remaja putri SMP/SMA di Indonesia.

Kepribadianmu:
- Empatik, hangat, sabar, dan TIDAK PERNAH menghakimi apa pun yang diceritakan pengguna.
- Menggunakan bahasa Indonesia santai namun sopan, sesuai usia remaja (13-18 tahun).
- Mendengarkan dulu sebelum memberi saran — jangan buru-buru menceramahi.

Batasan penting yang WAJIB kamu patuhi:
- Kamu BUKAN pengganti psikolog/psikiater/konselor manusia. Jangan pernah membuat diagnosis klinis.
- Jangan meminta atau menyimpan data pribadi sensitif (alamat lengkap, nomor identitas, dst).
- Kalau pengguna menunjukkan tanda risiko tinggi (indikasi kekerasan aktif, keinginan menyakiti \
diri sendiri, atau bahaya mendesak), WAJIB dengan tenang mendorong mereka menghubungi orang dewasa \
tepercaya (orang tua/guru BK) atau layanan bantuan resmi seperti SAPA 129, dan tetap menemani secara \
suportif — jangan mengabaikan atau meremehkan.
- Jangan membahas topik di luar cakupan dukungan emosional & edukasi pencegahan kekerasan.

Konteks emosi pengguna saat ini (hasil analisis internal, JANGAN disebut eksplisit ke pengguna,
cukup jadi pertimbangan nada bicaramu): {emotion_context}
"""

_EMOTION_CONTEXT_MAP: dict[EmotionLabel, str] = {
    "aman": "Pengguna tampak tenang/nyaman. Boleh lanjut dengan nada ringan dan suportif.",
    "netral": "Emosi pengguna netral. Tetap terbuka dan ajak bercerita lebih lanjut jika perlu.",
    "sedih": "Pengguna tampak sedih. Validasi perasaannya dulu, jangan buru-buru memberi solusi.",
    "takut": "Pengguna tampak takut/cemas. Tenangkan dulu, pastikan dia merasa aman bercerita.",
    "marah": "Pengguna tampak marah/frustrasi. Beri ruang, jangan defensif, dengarkan tanpa menghakimi.",
    "menyinggung": (
        "Teks mengandung indikasi konten menyinggung/berisiko (mis. kekerasan verbal, ejekan, atau "
        "situasi tidak nyaman). Tanggapi dengan hati-hati dan penuh empati, gali lebih lanjut secara halus."
    ),
}

# Label yang memicu eskalasi (ajakan ke bantuan profesional) di v1 —
# akan digantikan/diperhalus oleh model intent+darurat khusus begitu
# dataset "curhat | bertanya | minta_saran | minta_bantuan | darurat"
# di checklist tech stack selesai dilatih.
ESCALATION_LABELS: set[EmotionLabel] = {"takut", "menyinggung", "marah"}
ESCALATION_CONFIDENCE_THRESHOLD = 0.6


def should_escalate(label: EmotionLabel, confidence: float) -> bool:
    return label in ESCALATION_LABELS and confidence >= ESCALATION_CONFIDENCE_THRESHOLD


def _build_messages(
    text: str,
    history: list[ChatMessage],
    emotion_label: EmotionLabel,
) -> list[dict]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        emotion_context=_EMOTION_CONTEXT_MAP.get(
            emotion_label, "Tidak ada konteks emosi spesifik."
        )
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Riwayat percakapan sebelumnya (dibatasi 10 pesan terakhir supaya
    # prompt tidak membengkak / kena limit token Groq).
    for m in history[-10:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": text})
    return messages


class GroqCounselingClient:
    def __init__(self, api_key: str, model: str):
        self._model = model
        self._client: Groq | None = Groq(api_key=api_key) if api_key else None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def reply(
        self,
        text: str,
        history: list[ChatMessage],
        emotion_label: EmotionLabel,
    ) -> str:
        if not self._client:
            raise RuntimeError(
                "GROQ_API_KEY belum diisi di .env — lihat .env.example."
            )

        messages = _build_messages(text, history, emotion_label)

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()


_client_instance: GroqCounselingClient | None = None


def get_groq_client() -> GroqCounselingClient:
    global _client_instance
    if _client_instance is None:
        settings = get_settings()
        _client_instance = GroqCounselingClient(
            api_key=settings.groq_api_key, model=settings.groq_model
        )
    return _client_instance
