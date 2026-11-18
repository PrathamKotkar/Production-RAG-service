FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 && \
    pip install --no-cache-dir -r requirements.txt

# Bake models into the image so containers start fast and offline
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
