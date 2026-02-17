# Python 3.12
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# Librerías
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]