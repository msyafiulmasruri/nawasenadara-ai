"""
Loader & inferensi model klasifikasi 6 sentimen/emosi IndoBERT.

PENTING soal framework: model ini dilatih dengan PyTorch (lihat
notebook training — `transformers.Trainer` + `torch`, arsitektur
`BertForSequenceClassification`). Untuk tetap menyajikannya lewat
TensorFlow sesuai kebutuhan proyek, kita pakai `from_pt=True` saat
memuat model dengan kelas TF-nya — ini fitur resmi dari HuggingFace
`transformers` yang otomatis mengonversi bobot PyTorch (.bin/
.safetensors) menjadi tensor TensorFlow SEKALI saat proses start-up.
Tidak perlu training ulang, dan sesudah dimuat, seluruh proses
inferensi murni berjalan di atas graph TensorFlow.
"""

import logging
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("nawasenadara.sentiment_model")

# TensorFlow & transformers sengaja di-import di dalam try/except —
# keduanya lib besar (>1GB gabungan dengan torch). Kalau belum
# ter-install atau gagal load (mis. saat baru setup environment), modul
# ini TETAP bisa di-import tanpa crash; SentimentModel.load() akan
# melempar error yang jelas dan tertangkap di lifespan app/main.py,
# sehingga /health tetap bisa diakses untuk mendiagnosis masalahnya.
try:
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    _ML_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - tergantung environment
    tf = None
    AutoTokenizer = None
    TFAutoModelForSequenceClassification = None
    _ML_IMPORT_ERROR = exc


class SentimentModel:
    def __init__(self, model_dir: str, max_length: int = 128):
        self.model_dir = model_dir
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._id2label: dict[int, str] = {}

    def load(self) -> None:
        if _ML_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TensorFlow/transformers belum siap di environment ini "
                f"({_ML_IMPORT_ERROR}). Jalankan: pip install -r requirements.txt"
            )

        logger.info("Memuat tokenizer dari %s ...", self.model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        logger.info(
            "Memuat model (konversi bobot PyTorch -> TensorFlow, from_pt=True) ..."
        )
        self._model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_dir, from_pt=True
        )

        # id2label diambil dari config.json model itu sendiri (bukan
        # di-hardcode di sini), supaya urutan label SELALU sinkron
        # dengan urutan output logits model — kalau suatu saat model
        # dilatih ulang dengan urutan label berbeda, kode ini otomatis
        # ikut menyesuaikan tanpa perlu diedit manual.
        self._id2label = {
            int(k): v for k, v in self._model.config.id2label.items()
        }
        logger.info("Model siap. Label: %s", self._id2label)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if not self.is_loaded:
            raise RuntimeError(
                "Model belum dimuat. Pastikan startup event FastAPI sudah "
                "berjalan (lihat app/main.py)."
            )

        inputs = self._tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        outputs = self._model(**inputs)
        logits = outputs.logits.numpy()[0]
        probs = tf.nn.softmax(logits).numpy()

        scores = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }
        best_idx = int(np.argmax(probs))
        best_label = self._id2label[best_idx]
        best_confidence = float(probs[best_idx])

        return best_label, best_confidence, scores


@lru_cache
def get_sentiment_model() -> SentimentModel:
    settings = get_settings()
    model = SentimentModel(
        model_dir=settings.sentiment_model_dir,
        max_length=settings.max_sequence_length,
    )
    return model
