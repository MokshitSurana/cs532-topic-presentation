FROM python:3.10-slim

WORKDIR /app

# Install Git and basic packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch accelerate huggingface_hub python-dotenv
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
