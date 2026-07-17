FROM python:3.11-slim as builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install with extended timeout and retries for large binary packages
RUN pip install \
    --no-cache-dir \
    --default-timeout=600 \
    --retries=10 \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# System dependencies for runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
