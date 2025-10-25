FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy only requirements first (cache benefit)
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# now copy rest of project
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
