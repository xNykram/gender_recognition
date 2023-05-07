FROM python:3.11.3-slim

COPY src/ src/

COPY requirements.txt .

ENV PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
