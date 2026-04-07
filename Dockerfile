FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .
COPY . .
CMD ["uvicorn", "server.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
