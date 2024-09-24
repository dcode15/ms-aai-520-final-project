FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    WORKDIR="/usr/llm/src" \
    TRANSFORMERS_CACHE="/model_cache" \
    CUDA_HOME="/usr/local/cuda"

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $WORKDIR
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

RUN mkdir -p /model_cache && chmod 777 /model_cache

ENTRYPOINT ["echo", "'Pass an entrypoint from docker-compose.yml.'"]