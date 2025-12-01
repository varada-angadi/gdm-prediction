# ---------------------------
# 1. Base image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Install system dependencies (needed for ML & Tensorflow)
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 4. Copy project files
# ---------------------------
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

# ---------------------------
# 5. Install Python DEPENDENCIES
# ---------------------------
RUN pip install --no-cache-dir -r backend/requirements.txt

# ---------------------------
# 6. Render will assign a PORT env variable.
# ---------------------------
ENV PORT=10000

# ---------------------------
# 7. Run the app using Gunicorn (production server)
# ---------------------------
CMD gunicorn --bind 0.0.0.0:$PORT backend.app:app
