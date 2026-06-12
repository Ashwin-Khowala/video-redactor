FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[cpu]"

COPY src/ ./src/

RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

ENTRYPOINT ["video-redactor"]
