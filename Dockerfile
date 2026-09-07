FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY shopmind ./shopmind
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY configs ./configs

RUN python -m pip install --upgrade pip && python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "shopmind.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
