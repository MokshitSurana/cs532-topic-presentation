Great choice! 🚀 The **StableLM 2 1.6B (Quantized)** model is small enough to deploy efficiently on **Google Cloud Run**, and it works great for text generation tasks.

Here’s a full guide to build and deploy the quantized version of StableLM using:

- `main.py` (FastAPI app)
- `model.py` (model loading + prediction)
- `Dockerfile` (to containerize your app)
- ✅ Fully ready for GCP Cloud Run

---

## ✅ STEP 1: Project Structure

```
project-root/
│
├── main.py
├── model.py
├── Dockerfile
└── requirements.txt (optional)
```

---

## 📦 STEP 2: `model.py` – Load StableLM 2 1.6B (Quantized)

We'll use HuggingFace's quantized version:  
**`stabilityai/stablelm-2-1_6b-chat`** (GGUF or 4-bit quantized)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Load StableLM 2 1.6B (Quantized version)
model_name = "stabilityai/stablelm-2-1_6b-chat"

token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # works better for quantized models
    low_cpu_mem_usage=True,
    device_map="auto",
    token=token
)

# Create generation pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def answer_question(prompt: str) -> str:
    formatted = f"[INST] {prompt} [/INST]"
    result = qa_pipeline(formatted, max_new_tokens=100, do_sample=True)
    return result[0]["generated_text"].split("[/INST]")[-1].strip()
```

---

## 🌐 STEP 3: `main.py` – FastAPI Interface

```python
from fastapi import FastAPI
from pydantic import BaseModel
from model import answer_question

app = FastAPI()

class PromptRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: PromptRequest):
    response = answer_question(req.question)
    return {"question": req.question, "answer": response}
```

---

## 🐳 STEP 4: Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install Git and basic packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch accelerate huggingface_hub

# Set environment variable for Hugging Face Token
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 📄 `requirements.txt`

```txt
fastapi
uvicorn
transformers
torch
accelerate
huggingface_hub
```

---

## 🌐 STEP 5: Hugging Face Token

Create a `.env` file (locally only):

```bash
HF_TOKEN=your_huggingface_token_here
```

Then load it into your shell:

```bash
export $(cat .env | xargs)
```

---

## ☁️ STEP 6: Build and Deploy on GCP

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
```

### 🚧 Build Docker Image

```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t stablelm-api .
```

### 🚀 Push Image to GCP

```bash
docker tag stablelm-api gcr.io/YOUR_PROJECT_ID/stablelm-api
docker push gcr.io/YOUR_PROJECT_ID/stablelm-api
```

### 🔁 Deploy to Cloud Run

```bash
gcloud run deploy stablelm-api \
  --image gcr.io/YOUR_PROJECT_ID/stablelm-api \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars HF_TOKEN=$HF_TOKEN,HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
```

---

## 🧪 Test Your API

```bash
curl -X POST https://stablelm-api-xyz.a.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

✅ Response:
```json
{
  "question": "What is artificial intelligence?",
  "answer": "Artificial intelligence is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence..."
}
```

---
