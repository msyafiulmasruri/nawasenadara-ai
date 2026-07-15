"""
Skrip diagnosa MANDIRI (tidak lewat FastAPI/uvicorn sama sekali) untuk
melihat error ASLI kenapa model gagal dimuat.

KENAPA PERLU INI: `app/main.py` sengaja menangkap semua exception saat
model.load() gagal (lifespan startup) supaya server tetap bisa jalan dan
/health tetap bisa diakses untuk diagnosa — traceback lengkapnya CUMA
tercetak di log/terminal tempat `uvicorn` berjalan (lewat
`logger.exception(...)`), TIDAK dikirim ke response API. Jadi response
"Model NLP belum siap." di endpoint /analyze itu memang sengaja
digeneralisir, bukan error aslinya.

CARA PAKAI:
    cd nawasenadara-nlp-service   # root proyek, folder yang ada app/, model/, .env
    python diagnose_model.py

Kalau tidak mau ganggu .venv proyek, jalankan lewat python venv yang
sama dengan yang dipakai `uvicorn app.main:app` (aktifkan .venv dulu).
"""

import sys
import traceback

print("=" * 70)
print("DIAGNOSA MODEL NLP — Nawasena Dara")
print("=" * 70)

# 1. Cek working directory & apakah dijalankan dari root proyek yang benar
import os

cwd = os.getcwd()
print(f"\n[1] Current working directory : {cwd}")
if not os.path.isdir(os.path.join(cwd, "app")):
    print(
        "    !! PERINGATAN: folder 'app/' tidak ditemukan di sini.\n"
        "    Skrip ini HARUS dijalankan dari root proyek "
        "nawasenadara-nlp-service (folder yang berisi app/, model/, .env),\n"
        "    bukan dari dalam folder app/ atau folder lain."
    )

# 2. Load settings (baca .env) supaya tahu persis path model_dir yang
#    BENAR-BENAR dipakai (bukan cuma asumsi "./model")
try:
    from app.core.config import get_settings

    settings = get_settings()
    model_dir = settings.sentiment_model_dir
except Exception:
    print("\n[2] Gagal import app.core.config — pastikan dijalankan dari root proyek.")
    traceback.print_exc()
    sys.exit(1)

resolved_model_dir = os.path.abspath(model_dir)
print(f"\n[2] SENTIMENT_MODEL_DIR (dari .env) : {model_dir}")
print(f"    Resolusi absolut                 : {resolved_model_dir}")

if not os.path.isdir(resolved_model_dir):
    print(
        f"\n    !! FOLDER TIDAK DITEMUKAN: {resolved_model_dir}\n"
        "    Ini SANGAT MUNGKIN penyebab utamanya — cek lagi apakah folder\n"
        "    'model/' benar-benar ada persis di root proyek ini (sejajar\n"
        "    dengan folder app/), atau kalau kamu set path custom di .env,\n"
        "    pastikan path itu benar relatif terhadap folder tempat kamu\n"
        "    menjalankan `uvicorn app.main:app` (BUKAN relatif terhadap\n"
        "    lokasi file .env atau app/ itu sendiri)."
    )
    sys.exit(1)

# 3. List isi folder model — pastikan model.safetensors ATAU
#    pytorch_model.bin benar-benar ada dan bukan cuma config/tokenizer
print(f"\n[3] Isi folder model ({resolved_model_dir}):")
found_weights = False
for fname in sorted(os.listdir(resolved_model_dir)):
    fpath = os.path.join(resolved_model_dir, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"    - {fname:<30} {size_mb:>10.2f} MB")
    if fname in ("model.safetensors", "pytorch_model.bin", "tf_model.h5"):
        found_weights = True

if not found_weights:
    print(
        "\n    !! TIDAK ADA FILE BOBOT MODEL (model.safetensors / "
        "pytorch_model.bin) di folder ini.\n"
        "    Cuma ada config.json/tokenizer — ini PENYEBAB PALING UMUM.\n"
        "    Taruh file model.safetensors (~400MB, hasil trainer.save_model())\n"
        "    persis di folder ini, dengan nama file itu juga (jangan diganti\n"
        "    nama)."
    )
    sys.exit(1)

# 4. Cek konsistensi config.json (num_labels vs id2label) — bug yang
#    sudah ditemukan & diperbaiki terpisah, tapi dicek ulang di sini
#    supaya ketahuan kalau file yang dipakai masih versi lama.
import json

config_path = os.path.join(resolved_model_dir, "config.json")
if os.path.isfile(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    n_declared = cfg.get("_num_labels")
    n_actual = len(cfg.get("id2label", {}))
    print(f"\n[4] config.json: _num_labels={n_declared}, jumlah id2label={n_actual}")
    if n_declared is not None and n_declared != n_actual:
        print(
            f"    !! MISMATCH: _num_labels ({n_declared}) != jumlah id2label "
            f"({n_actual}).\n"
            "    Ini bisa bikin classifier head model salah ukuran saat "
            "dimuat/dikonversi dari PyTorch.\n"
            "    FIX: ganti '_num_labels' di config.json jadi sama dengan "
            f"jumlah id2label ({n_actual})."
        )

# 5. Cek import TensorFlow/transformers/torch satu-satu (bukan sekaligus)
#    supaya kalau salah satu gagal, jelas modul MANA yang bermasalah.
print("\n[5] Cek import library satu per satu:")
for modname in ("torch", "tensorflow", "transformers"):
    try:
        mod = __import__(modname)
        version = getattr(mod, "__version__", "?")
        print(f"    - {modname:<15} OK (versi {version})")
    except Exception as exc:
        print(f"    - {modname:<15} GAGAL: {exc!r}")
        print(
            f"\n    !! '{modname}' gagal di-import. Jalankan:\n"
            f"       pip install -r requirements.txt\n"
            "    (pastikan .venv proyek ini yang aktif, bukan Python global)"
        )
        sys.exit(1)

# 6. Percobaan SESUNGGUHNYA memuat model — traceback LENGKAP akan
#    tercetak di sini kalau masih gagal, sekarang tidak ditangkap
#    diam-diam seperti di lifespan FastAPI.
print("\n[6] Mencoba memuat tokenizer + model (ini yang paling penting)...\n")
try:
    from app.services.sentiment_model import SentimentModel

    model = SentimentModel(model_dir=resolved_model_dir, max_length=settings.max_sequence_length)
    model.load()
    print("\n✅ BERHASIL — model berhasil dimuat sepenuhnya.")

    label, confidence, scores = model.predict("halo")
    print(f"\nTes prediksi teks 'halo':")
    print(f"    label      : {label}")
    print(f"    confidence : {confidence:.4f}")
    print(f"    scores     : {scores}")
except Exception:
    print("\n❌ GAGAL memuat model. Traceback LENGKAP di bawah ini:\n")
    traceback.print_exc()
    print(
        "\n" + "=" * 70 + "\n"
        "Salin traceback di atas apa adanya kalau butuh bantuan lebih "
        "lanjut — pesan error paling bawah (baris terakhir sebelum akhir\n"
        "traceback) biasanya paling menunjukkan akar masalahnya."
    )
    sys.exit(1)

print("\n" + "=" * 70)
print("Semua pengecekan lolos. Restart server (`uvicorn app.main:app "
      "--reload`) sekarang seharusnya membuat /health -> model_loaded: true.")
print("=" * 70)
