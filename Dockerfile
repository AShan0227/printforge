FROM python:3.11-slim

# System dependencies for trimesh rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY web/ web/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "printforge.server:app", "--host", "0.0.0.0", "--port", "8000"]
