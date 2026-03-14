FROM python:3.12-slim

WORKDIR /opt/recsys

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "cart_driven_recsys.api.main:app", "--host", "0.0.0.0", "--port", "8000"]