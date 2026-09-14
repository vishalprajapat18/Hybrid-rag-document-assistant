FROM python:3.13-slim

# Hugging Face Spaces runs the container as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    API_BASE_URL=http://127.0.0.1:8000

WORKDIR $HOME/app

# CPU-only torch first. The default wheel bundles CUDA libraries (~2.5 GB) that a CPU machine cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding and reranking models at build time, not on every container start
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY --chown=user . .

EXPOSE 7860

CMD ["bash", "start.sh"]
