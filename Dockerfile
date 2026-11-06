# MammoRisk+ platform (O5). Build:  docker compose build   Run:  docker compose up
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/mammorisk

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Only the code the platform needs. No data, genomes, images or weights are copied into the image;
# model artefacts are mounted at run time (see docker-compose.yml).
COPY app/ app/
COPY configs/app.yaml configs/app.yaml

RUN useradd --create-home --uid 10001 mammorisk \
    && mkdir -p /tmp/mammorisk-uploads \
    && chown mammorisk /tmp/mammorisk-uploads
USER mammorisk

EXPOSE 8000
# SECRET_KEY and user password hashes come from the environment, never from the image.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "app:create_app()"]
