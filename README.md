# RAG Document Assistant

> Chat với tài liệu của bạn — hỏi đáp thông minh dựa trên nội dung PDF, DOCX, TXT, Markdown bằng Retrieval-Augmented Generation.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Tech Stack](#tech-stack)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Chạy ứng dụng](#chạy-ứng-dụng)
  - [Streamlit UI](#1-streamlit-ui-giao-diện-chat)
  - [FastAPI Backend](#2-fastapi-backend-rest-api)
  - [Docker Compose](#3-docker-compose-đầy-đủ)
- [API Reference](#api-reference)
- [Kiểm thử](#kiểm-thử)
- [Đánh giá chất lượng](#đánh-giá-chất-lượng-ragas)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Giới thiệu

**RAG Document Assistant** là ứng dụng web cho phép bạn tải lên tài liệu và đặt câu hỏi bằng ngôn ngữ tự nhiên. Hệ thống sẽ:

1. Phân tách tài liệu thành các đoạn (chunks)
2. Chuyển đổi chúng thành vector embedding
3. Lưu trữ vào ChromaDB
4. Khi có câu hỏi → tìm các chunk liên quan nhất → gửi vào LLM (Claude/GPT/Llama) → trả lời có trích dẫn nguồn

### Vì sao dự án này hữu ích?

- Hỏi đáp nhanh trên tài liệu dài (báo cáo, sách, giáo trình)
- Câu trả lời luôn **có trích dẫn nguồn** (tên file, trang)
- Chạy local được hoàn toàn với Ollama (miễn phí, không cần API key)
- Kiến trúc RAG chuẩn — phù hợp làm portfolio AI Engineer

---

## Tính năng

- Hỗ trợ nhiều định dạng: **PDF, DOCX, TXT, Markdown**
- Chat UI với Streamlit — upload nhiều file cùng lúc
- REST API với FastAPI — auto-generate Swagger docs tại `/docs`
- Multi-provider LLM: **Anthropic Claude**, **OpenAI GPT**, **Ollama** (local)
- Embedding chạy local miễn phí (`all-MiniLM-L6-v2`)
- Trích dẫn nguồn đầy đủ (file, trang, chunk index)
- Conversation memory cho hỏi đáp nhiều lượt
- Evaluation pipeline với RAGAS
- Docker-ready với docker-compose

---

## Kiến trúc hệ thống

```
                              RAG Pipeline

  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌────────────┐
  │ Document │───>│ Chunking  │───>│ Embedding  │───>│ Vector DB  │
  │ Upload   │    │ (Split +  │    │ (MiniLM    │    │ (ChromaDB) │
  │          │    │  Overlap) │    │  L6-v2)    │    │            │
  └──────────┘    └───────────┘    └────────────┘    └─────┬──────┘
                                                           │
                                                           │ top-k
  ┌──────────┐    ┌───────────┐    ┌────────────┐          │
  │ Response │<───│    LLM    │<───│  Retrieval │<─────────┘
  │ + Source │    │ (Claude/  │    │ (Similarity│<─── User Query
  │ Citation │    │ GPT/Llama)│    │  Search)   │
  └──────────┘    └───────────┘    └────────────┘
```

### Data flow chi tiết

1. **Ingestion** — [src/ingestion/loader.py](src/ingestion/loader.py) đọc file bằng LangChain loaders.
2. **Chunking** — [src/ingestion/chunker.py](src/ingestion/chunker.py) chia nhỏ với `RecursiveCharacterTextSplitter` (1000 ký tự, overlap 200).
3. **Embedding** — [src/embeddings/store.py](src/embeddings/store.py) dùng HuggingFace `all-MiniLM-L6-v2` → vector 384 chiều.
4. **Retrieval** — [src/retrieval/retriever.py](src/retrieval/retriever.py) tìm top-k chunks gần nhất (cosine similarity).
5. **Generation** — [src/generation/rag_chain.py](src/generation/rag_chain.py) xây LangChain pipeline: `query → retrieve → format context → LLM → answer`.

---

## Tech Stack

| Thành phần       | Công nghệ                       | Ghi chú                               |
| ---------------- | ------------------------------- | ------------------------------------- |
| Framework        | LangChain 0.3+                  | Orchestration cho RAG                 |
| Vector DB        | ChromaDB                        | Lưu trữ embedding local               |
| LLM              | Claude / GPT / Gemini / Groq / Ollama | Switch qua biến `LLM_PROVIDER`  |
| Embedding        | `all-MiniLM-L6-v2`              | HuggingFace — chạy local, miễn phí    |
| Backend          | FastAPI + Uvicorn               | REST API với auto-docs                |
| Frontend         | Streamlit                       | Chat UI nhanh chóng                   |
| Container        | Docker + Docker Compose         | Deploy dễ dàng                        |
| Evaluation       | RAGAS                           | Đo faithfulness, relevancy, precision |

---

## Cấu trúc thư mục

```
rag-document-assistant/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # Đọc PDF, DOCX, TXT, MD
│   │   └── chunker.py         # Tách chunks với overlap
│   ├── embeddings/
│   │   └── store.py           # Embedding + ChromaDB
│   ├── retrieval/
│   │   └── retriever.py       # Vector similarity search
│   ├── generation/
│   │   ├── rag_chain.py       # Core RAG chain
│   │   └── memory.py          # Conversation memory
│   └── api/
│       └── routes.py          # FastAPI endpoints
├── app/
│   └── main.py                # Streamlit UI
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_store.py
│   ├── test_rag_chain.py
│   └── test_evaluation.py     # RAGAS evaluation
├── data/
│   ├── samples/               # Tài liệu mẫu
│   └── uploads/               # File upload runtime
├── conftest.py                # Pytest config
├── run_api.py                 # Entry point FastAPI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Yêu cầu hệ thống

- **Python** >= 3.11
- **pip** >= 23
- (Khuyến nghị) **Git** để clone repo
- (Tuỳ chọn) **Docker** >= 24 + **Docker Compose** >= 2
- **RAM**: tối thiểu 4GB (embedding model ~90MB)
- **Disk**: ~2GB cho dependencies + model cache
- **API key**: Anthropic hoặc OpenAI (hoặc cài Ollama để chạy local free)

---

## Cài đặt

### Bước 1 — Clone và vào thư mục dự án

```bash
git clone <your-repo-url> rag-document-assistant
cd rag-document-assistant
```

### Bước 2 — Tạo virtual environment

**Windows — CMD (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Nếu PowerShell báo lỗi chính sách thực thi, chạy:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Windows — Git Bash:**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Bước 3 — Cài dependencies

**Cài full (bao gồm dev + eval):**
```bash
pip install -e ".[dev,eval]"
```

**Hoặc cài tối thiểu để chạy app:**
```bash
pip install -e .
```

**Thêm provider tuỳ chọn:**
```bash
pip install -e ".[openai]"    # nếu dùng OpenAI
pip install -e ".[ollama]"    # nếu dùng Ollama
```

### Bước 4 — Pre-download embedding model (tuỳ chọn)

Để tránh chờ lần đầu chạy:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Cấu hình

### Tạo file `.env`

**Windows CMD:**
```cmd
copy .env.example .env
```

**PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Git Bash / macOS / Linux:**
```bash
cp .env.example .env
```

> Lưu ý: File `.env.example` là **dot-file**, có thể bị ẩn trong File Explorer.
> Bật hiển thị bằng: **View → Show → Hidden items** (hoặc trong VS Code mặc định đã hiện).

### Bảng so sánh nhanh các provider

| Provider | Free? | Tốc độ | Chất lượng | Khuyến nghị cho |
|---|---|---|---|---|
| **Gemini**    | ✅ Free tier (15 req/min, 1500/ngày) | Nhanh    | ⭐⭐⭐⭐  | **Học, demo nhỏ** |
| **Groq**      | ✅ Free tier rất generous           | **Cực nhanh** (>500 t/s) | ⭐⭐⭐⭐ | **Demo realtime** |
| **Anthropic** | ❌ Pay-as-you-go (~$5 nạp tối thiểu) | Nhanh    | ⭐⭐⭐⭐⭐ | Production    |
| **OpenAI**    | ❌ Pay-as-you-go                    | Nhanh    | ⭐⭐⭐⭐⭐ | Production    |
| **Ollama**    | ✅ 100% free (local)                | Chậm trên CPU | ⭐⭐⭐ | Privacy, offline |

### Chỉnh sửa `.env` theo provider bạn chọn

#### Option A — Google Gemini (FREE, khuyến nghị cho học)

1. Lấy API key (miễn phí, chỉ cần Google account):
   https://aistudio.google.com/apikey
2. Cài adapter:
   ```cmd
   pip install -e ".[gemini]"
   ```
3. Cấu hình `.env`:
   ```env
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=AIza...your-key...
   ```

**Free tier limits:**
- 15 requests/phút
- 1,500 requests/ngày
- 1M tokens/phút

**Model mặc định**: `gemini-2.0-flash`. Đổi qua `LLM_MODEL=gemini-1.5-flash` nếu muốn ổn định hơn.

#### Option B — Groq (FREE, nhanh nhất)

Groq dùng LPU chip, tốc độ inference cực cao (>500 tokens/giây).

1. Lấy API key:
   https://console.groq.com/keys
2. Cài adapter:
   ```cmd
   pip install -e ".[groq]"
   ```
3. Cấu hình `.env`:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_...your-key...
   ```

**Free tier limits** (per model, khá thoáng):
- 30 requests/phút
- 14,400 requests/ngày

**Model mặc định**: `llama-3.3-70b-versatile`. Các option khác:
- `llama-3.1-8b-instant` — nhanh nhất
- `mixtral-8x7b-32768` — context dài 32K
- `gemma2-9b-it` — Google open model

Đổi qua biến `LLM_MODEL`:
```env
LLM_MODEL=llama-3.1-8b-instant
```

#### Option C — Anthropic Claude (TRẢ PHÍ, chất lượng cao nhất)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...your-key...
```

Lấy API key: https://console.anthropic.com/ (cần nạp tối thiểu $5)

#### Option D — OpenAI GPT (TRẢ PHÍ)

```cmd
pip install -e ".[openai]"
```

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your-key...
```

Model mặc định: `gpt-4o-mini` (rẻ). Đổi `LLM_MODEL=gpt-4o` cho chất lượng cao hơn.

#### Option E — Ollama (LOCAL, miễn phí nhưng tốn tài nguyên)

1. Cài Ollama: https://ollama.com/download
2. Pull model (chọn theo RAM máy):
   ```cmd
   ollama pull gemma2:2b      :: máy 8GB RAM
   ollama pull llama3         :: máy 16GB RAM
   ```
3. Cài adapter:
   ```cmd
   pip install -e ".[ollama]"
   ```
4. Cấu hình `.env`:
   ```env
   LLM_PROVIDER=ollama
   LLM_MODEL=gemma2:2b
   ```

### Cài tất cả provider cùng lúc

```cmd
pip install -e ".[all-providers]"
```

### Các biến môi trường khác

| Biến                     | Mặc định           | Mô tả                                                        |
| ------------------------ | ------------------ | ------------------------------------------------------------ |
| `LLM_PROVIDER`           | `anthropic`        | `anthropic` / `openai` / `gemini` / `groq` / `ollama`        |
| `LLM_MODEL`              | (theo provider)    | Override model mặc định của provider                         |
| `EMBEDDING_MODEL`        | `all-MiniLM-L6-v2` | HuggingFace embedding model                                  |
| `CHROMA_PERSIST_DIR`     | `.chroma`          | Thư mục lưu ChromaDB                                         |
| `CHROMA_COLLECTION_NAME` | `documents`        | Tên collection trong ChromaDB                                |

---

## Chạy ứng dụng

### 1. Streamlit UI (giao diện chat)

```bash
streamlit run app/main.py
```

Mở trình duyệt tại http://localhost:8501

**Cách dùng:**
1. Upload file (PDF / DOCX / TXT / MD) ở sidebar
2. Nhấn **"Xử lý tài liệu"**
3. Gõ câu hỏi vào ô chat
4. Xem câu trả lời + nguồn trích dẫn trong expander "Nguồn trích dẫn"

### 2. FastAPI Backend (REST API)

**Cách 1 — Dùng script entry point:**
```bash
python run_api.py
```

**Cách 2 — Dùng uvicorn trực tiếp:**
```bash
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

- API docs (Swagger): http://localhost:8000/docs
- API docs (ReDoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### 3. Docker Compose (đầy đủ)

```bash
docker compose up --build
```

- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000

**Dừng service:**
```bash
docker compose down
```

**Xoá data (bao gồm ChromaDB):**
```bash
docker compose down -v
```

---

## API Reference

### `POST /upload`

Upload và xử lý tài liệu.

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@path/to/document.pdf"
```

**Response:**
```json
{
  "message": "Processed 42 chunks from document.pdf"
}
```

### `POST /ask`

Đặt câu hỏi về tài liệu đã upload.

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tóm tắt nội dung chính?", "k": 4}'
```

**Response:**
```json
{
  "answer": "Tài liệu nói về...",
  "sources": [
    {
      "file": "document.pdf",
      "page": 3,
      "content_preview": "..."
    }
  ]
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

## Kiểm thử

### Chạy toàn bộ tests

```bash
pytest tests/ -v
```

### Chạy với coverage report

```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

Mở `htmlcov/index.html` để xem báo cáo chi tiết.

### Chạy tests cụ thể

```bash
pytest tests/test_loader.py -v           # chỉ test loader
pytest tests/test_chunker.py::test_chunk_long_document -v   # chỉ 1 test
```

### Linting & formatting

```bash
ruff check .
black --check .
black .    # auto-fix formatting
```

---

## Đánh giá chất lượng (RAGAS)

Sau khi cài extra `eval`:
```bash
pip install -e ".[eval]"
```

Chạy đánh giá:

```python
from src.embeddings.store import load_vector_store
from tests.test_evaluation import evaluate_rag

store = load_vector_store()

questions = ["Câu hỏi 1?", "Câu hỏi 2?"]
ground_truths = ["Đáp án đúng 1", "Đáp án đúng 2"]

results = evaluate_rag(questions, ground_truths, store)
print(results)
```

Các metric:
- **Faithfulness** — câu trả lời có bám sát context không
- **Answer Relevancy** — câu trả lời có đúng trọng tâm câu hỏi không
- **Context Precision** — retrieval có chính xác không

---

## Troubleshooting

### 1. `ImportError: No module named 'src'`

Đảm bảo chạy lệnh từ **thư mục gốc** của dự án:
```bash
cd rag-document-assistant
streamlit run app/main.py    # chạy từ đây, không phải từ app/
```

### 2. `ANTHROPIC_API_KEY not set`

Check file `.env` có tồn tại và đúng định dạng:
```bash
cat .env
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Lỗi khi load PDF có OCR kém

Dùng PDF text-based thay vì scanned PDF. Nếu buộc phải xử lý scanned PDF, cần thêm bước OCR (Tesseract).

### 4. Embedding model download chậm lần đầu

Lần đầu chạy sẽ tải ~90MB model. Các lần sau sẽ dùng cache tại `~/.cache/huggingface/`.

### 5. Docker build chậm

Sử dụng BuildKit để cache layer:
```bash
DOCKER_BUILDKIT=1 docker compose build
```

### 6. ChromaDB lock error

Nếu gặp lỗi ChromaDB bị lock, xoá thư mục `.chroma/` và upload lại tài liệu:
```bash
rm -rf .chroma
```

### 7. LangChain deprecation warnings

Các cảnh báo deprecation từ `langchain_community` không ảnh hưởng chức năng. Sẽ được migrate dần sang các package riêng (`langchain-chroma`, `langchain-huggingface`).

---

## Roadmap

- [x] Multi-format document loading (PDF, DOCX, TXT, MD)
- [x] RAG pipeline với citation tracking
- [x] Streamlit UI
- [x] FastAPI backend
- [x] Docker containerization
- [x] Unit tests
- [ ] Hybrid search (BM25 + vector)
- [ ] Re-ranking với Cohere/Cross-encoder
- [ ] Streaming response
- [ ] Multi-user với authentication
- [ ] Deploy lên HuggingFace Spaces / Railway

---

## License

MIT License — xem file [LICENSE](LICENSE) để biết chi tiết.

---

## Tài nguyên học thêm

- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [RAGAS Evaluation](https://docs.ragas.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Anthropic Claude API](https://docs.anthropic.com/)
