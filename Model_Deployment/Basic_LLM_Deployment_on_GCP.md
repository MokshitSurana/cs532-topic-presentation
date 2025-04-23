Deploying a basic next-word prediction model on **Google Cloud using FastAPI**:

---

## ✅ Overview of the Process

1. **Train or load a simple model** (e.g., using a tokenizer + n-gram or pre-trained language model).
2. **Create a FastAPI app** to serve predictions.
3. **Containerize the app** using Docker.
4. **Deploy it on Google Cloud Run**.

---

## 🧠 Step 1: Create a Simple Next-Word Prediction Model

We'll use a pre-trained model (`distilgpt2`) from Hugging Face.

```bash copy
pip install fastapi uvicorn transformers torch
```

### `model.py`

```python copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def predict_next_word(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1)
    next_token = outputs[0][-1].item()
    next_word = tokenizer.decode([next_token])
    return next_word.strip()
```

---

## 🌐 Step 2: Create a FastAPI App

### `main.py`

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import predict_next_word

app = FastAPI()

class PromptRequest(BaseModel):
    text: str

@app.post("/predict")
def get_next_word(data: PromptRequest):
    next_word = predict_next_word(data.text)
    return {"input": data.text, "next_word": next_word}
```

### Run locally (for testing):

```bash
uvicorn main:app --reload
```

---

## 🐳 Step 3: Dockerize the App

### `Dockerfile`

```Dockerfile
# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch

# Expose FastAPI port
EXPOSE 8080

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## ☁️ Step 4: Deploy to Google Cloud Run

### 🔧 4.1: Setup GCP (one-time)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
```

### 🔧 4.2: Build Docker Image with Cloud Build

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/nextword-api
```

### 🚀 4.3: Deploy to Cloud Run

```bash
gcloud run deploy nextword-api \
  --image gcr.io/YOUR_PROJECT_ID/nextword-api \
  --platform managed \
  --allow-unauthenticated
```

---

## 🧪 Step 5: Test It!

Once deployed, you’ll get a public URL like:

```
https://nextword-api-xyz.a.run.app
```

You can test it with:

```bash
curl -X POST https://nextword-api-xyz.a.run.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love"}'
```

Expected response:

```json
{
    "input": "I love",
    "next_word": "you" // or similar
}
```

---
