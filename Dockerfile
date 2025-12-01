FROM python:3.10-slim

WORKDIR /app

# Copy backend
COPY backend/ ./backend/

# Copy frontend (optional if serving via Flask)
COPY frontend ./frontend

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port (optional)
EXPOSE 5000

# Railway will inject PORT at runtime
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} backend.app:app"]
