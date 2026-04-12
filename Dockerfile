FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user

RUN useradd -m -u 1000 user

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY --chown=user:user . .

RUN uv sync --frozen --no-dev --no-editable && \
    rm -rf chaosagent.egg-info build dist

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

USER user

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
