FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy backend files directly
COPY backend/* .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
