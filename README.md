# Nawasena Dara NLP & Chatbot Konseling Service (FastAPI)

Microservice Python yang menyajikan:
1. **`POST /analyze`** - analisis sentimen/emosi teks bebas pemain, pakai model IndoBERT (6 label: `aman`, `menyinggung`, `takut`, `marah`, `netral`, `sedih`) yang sudah kamu latih, disajikan lewat **TensorFlow**.
2. **`POST /chat/counseling`** - endpoint gabungan: analisis emosi teks dulu (internal), lalu hasilnya jadi konteks untuk memanggil **Groq API (Llama)** yang berperan sebagai persona "Kak Dara", chatbot konseling virtual.

Service ini **tidak untuk diakses langsung oleh frontend/pemain** - dipanggil servertoserver oleh `nawasenadarabackend` (Express), yang lalu meneruskan hasilnya ke frontend Next.js.



## Catatan penting sebelum mulai

**Model dilatih dengan PyTorch, bukan TensorFlow.** Notebook training menggunakan `transformers.Trainer` + `torch`, dan `config.json` menyebut arsitektur `BertForSequenceClassification` (kelas PyTorch). Supaya tetap bisa disajikan lewat TensorFlow tanpa retraining, kode di sini memuat model dengan:

```python
TFAutoModelForSequenceClassification.from_pretrained(model_dir, from_pt=True)
```

`from_pt=True` adalah fitur resmi HuggingFace `transformers` yang otomatis mengonversi bobot PyTorch menjadi tensor TensorFlow **sekali saat startup**. Konsekuensinya: `torch` tetap harus terinstall di environment (dipakai sekali untuk membaca file bobot), meski proses inferensi sepenuhnya jalan di TensorFlow setelah model selesai dimuat.

**File bobot model belum ada di folder ini.** Yang sudah tersedia di `model/`: `config.json`, `tokenizer.json`, `tokenizer_config.json`. Kamu perlu menaruh `model.safetensors` (boleh sertakan juga `training_args.bin`, meski itu cuma metadata training, tidak dipakai saat inferensi) di folder yang sama - dari notebook training kamu, itu ada di:
```
/content/drive/MyDrive/Colab/Nawasena Dara/Intent_Classifier/
```
Cari file `model.safetensors` di folder itu (hasil dari `trainer.save_model(...)`), lalu taruh ke `model/` di proyek ini.



## Langkahlangkah di VS Code

### 1. Buka folder proyek
```bash
code nawasenadaranlpservice
```

### 2. Buat virtual environment Python
Buka terminal di VS Code (`` Ctrl+` ``):
```bash
python3 -m venv .venv
```
atau

```bash
py -m venv .venv
```

Aktifkan:
 **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
 **macOS/Linux:** `source .venv/bin/activate`

Pastikan VS Code memakai interpreter dari `.venv` ini: `Ctrl+Shift+P` **"Python: Select Interpreter"** pilih yang pathnya mengandung `.venv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Ini termasuk `tensorflowcpu`, `torch`, dan `transformers` - ukurannya besar (>1GB gabungan), proses install bisa makan beberapa menit.

### 4. Taruh file bobot model
Salin `model.safetensors` hasil training kamu ke folder `model/`. Sehingga isi foldernya jadi:
```
model/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors        < yang kamu tambahkan
  training_args.bin        < opsional, tidak dipakai saat inferensi
```

### 5. Siapkan file `.env`
```bash
cp .env.example .env
```
Isi minimal yang WAJIB diubah:
 `GROQ_API_KEY` - ambil gratis di https://console.groq.com/keys
 `INTERNAL_API_KEY` - string acak bebas (mis. `openssl rand hex 32`), harus SAMA dengan yang nanti diisi di `.env` backend Express saat memanggil service ini.

### 6. Jalankan server (development)
```bash
uvicorn app.main:app --reload --port 8001
```
Buka http://localhost:8001/docs - Swagger UI otomatis muncul, bisa langsung dites dari browser.

### 7. Cek health check
```bash
curl http://localhost:8001/health
```
Kalau `model_loaded: true` dan `groq_configured: true`, semuanya siap.

### 8. Tes endpoint analisis sentimen
```bash
curl X POST http://localhost:8001/analyze \
  H "ContentType: application/json" \
  H "XInternalApiKey: <isi sama dengan INTERNAL_API_KEY di .env>" \
  d '{"text": "aku takut banget sama dia, dia selalu ngancem aku"}'
```

### 9. Tes endpoint chatbot konseling (NLP + Groq)
```bash
curl X POST http://localhost:8001/chat/counseling \
  H "ContentType: application/json" \
  H "XInternalApiKey: <isi sama dengan INTERNAL_API_KEY di .env>" \
  d '{"text": "aku gatau harus cerita ke siapa, aku takut"}'
```

### 10. Jalankan automated test
```bash
pytest
```



## Integrasi ke `nawasenadarabackend` (Express)

Di backend Express, panggil service ini servertoserver (contoh pakai `fetch` bawaan Node 18+):

```js
// src/services/nlp/nlpclient.js
const NLP_SERVICE_URL = process.env.NLP_SERVICE_URL || 'http://localhost:8001';
const INTERNAL_API_KEY = process.env.NLP_INTERNAL_API_KEY;

export const analyzeSentiment = async (text) => {
  const res = await fetch(`${NLP_SERVICE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'ContentType': 'application/json',
      'XInternalApiKey': INTERNAL_API_KEY,
    },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error('NLP service error');
  return res.json(); // { label, confidence, scores }
};

export const chatCounseling = async (text, history = []) => {
  const res = await fetch(`${NLP_SERVICE_URL}/chat/counseling`, {
    method: 'POST',
    headers: {
      'ContentType': 'application/json',
      'XInternalApiKey': INTERNAL_API_KEY,
    },
    body: JSON.stringify({ text, history }),
  });
  if (!res.ok) throw new Error('NLP service error');
  return res.json(); // { reply, emotion_detected, emotion_confidence, risk_level, escalated }
};
```

Tambahkan ke `.env` backend Express:
```
NLP_SERVICE_URL=http://localhost:8001
NLP_INTERNAL_API_KEY=<harus SAMA dengan INTERNAL_API_KEY di .env service NLP ini>
```

Alur endtoend sesuai proposal (Gambar 3.1 & 3.2): pemain mengetik curhat di jurnal refleksi akhir episode / chatbot Express memanggil `chatCounseling()` di atas dapat balasan + `risk_level` + `escalated` Express simpan ke tabel `nlp_analysis` (PostgreSQL) kalau `escalated: true`, Express trigger notifikasi realtime (Socket.io) ke dashboard guru BK.



## Struktur Proyek

```
app/
  core/
    config.py       # Settings dari .env (pydanticsettings)
    security.py     # Verifikasi header XInternalApiKey
  services/
    sentiment_model.py   # Load & inferensi model IndoBERT via TensorFlow
    groq_client.py        # Persona konselor virtual + panggilan Groq API
    counseling.py         # Orkestrasi: analisis emosi > konteks > Groq
  routers/
    analyze.py       # POST /analyze
    chat.py           # POST /chat/counseling
  schemas.py          # Pydantic request/response models
  main.py             # Entry point, load model saat startup
model/
  config.json, tokenizer*.json + (kamu tambahkan) model.safetensors
tests/
  test_health.py
requirements.txt
.env.example
```
