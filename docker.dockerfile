# ---------------------------
# 1. Base image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Install system dependencies (if needed for ML libs)
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 4. Copy backend and frontend
# ---------------------------
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

# ---------------------------
# 5. Install Python dependencies
# ---------------------------
RUN pip install --no-cache-dir -r backend/requirements.txt

# ---------------------------
# 6. Expose Render port
# ---------------------------
EXPOSE 10000

# ---------------------------
# 7. Run app
# ---------------------------
CMD ["python", "backend/app.py"]
