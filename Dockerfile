# Imagem do servidor hybrid-rag-mcp (transporte HTTP).
# Build: docker build -t hybrid-rag-mcp .
# Uso típico: via docker-compose.yml (sobe Qdrant + RAG juntos).
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "hybrid_rag_mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
