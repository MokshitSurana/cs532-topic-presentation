# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch

# Expose FastAPI port (use 8080!)
EXPOSE 8080

# Run the app on Cloud Run’s required port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
