FROM node:22-bookworm AS frontend

WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY problems ./problems
COPY --from=frontend /src/dist ./frontend/dist

ENV LEETCV_DATA=/data
ENV PYTHONUNBUFFERED=1

EXPOSE 80
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "80"]
