FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured / pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache deps layer: install deps from pyproject.toml BEFORE copying source
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "langchain>=0.3" \
        "langchain-community>=0.3" \
        "langchain-anthropic>=0.3" \
        "langchain-huggingface>=0.1" \
        "langchain-chroma>=0.1" \
        "chromadb>=0.5" \
        "sentence-transformers>=2.7" \
        "pypdf>=4.0" \
        "python-docx>=1.1" \
        "unstructured>=0.14" \
        "markdown>=3.6" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "streamlit>=1.34" \
        "python-dotenv>=1.0" \
        "python-multipart>=0.0.9" && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
