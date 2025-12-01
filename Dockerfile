FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy backend files directly
COPY backend/* .
COPY frontend ./frontend

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start Gunicorn using Railway PORT
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]
