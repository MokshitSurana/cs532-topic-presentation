# 🚀 Deploying a Next-Word Prediction Model on **AWS (Elastic Beanstalk) using FastAPI**

---

## ✅ Overview of the Process

1. **Load a pre-trained language model** (e.g., distilgpt2).
2. **Build a FastAPI app** for serving next-word predictions.
3. **Package the app for AWS Elastic Beanstalk** using `Docker`.
4. **Deploy to AWS Elastic Beanstalk** using the AWS CLI.

---

## 🧠 Step 1: Create a Next-Word Prediction Model

We'll use a Hugging Face model (`distilgpt2`) and serve predictions via FastAPI.

```bash
pip install fastapi uvicorn transformers torch
```

### `model.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
from fastapi import FastAPI
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

### Test locally

```bash
uvicorn main:app --reload
```

---

## 🐳 Step 3: Dockerize the App

### `Dockerfile`

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn transformers torch

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 📁 Step 4: Prepare AWS Deployment Files

### `.ebextensions/python.config`

```yaml
option_settings:
    aws:elasticbeanstalk:container:python:
        WSGIPath: main:app
```

> Even though we’re using Docker, this file can be useful for certain Elastic Beanstalk configurations. But it's optional for a Docker-based deployment.

### `Dockerrun.aws.json` (optional if using Dockerfile)

If you're not using `Dockerrun.aws.json`, Elastic Beanstalk will use your `Dockerfile` by default.

---

## ☁️ Step 5: Deploy on AWS Elastic Beanstalk

### 🔧 5.1: Initialize Elastic Beanstalk

```bash
aws configure
eb init -p docker nextword-api --region us-east-1
```

-   Choose your application name (`nextword-api`)
-   Select the appropriate region
-   Create an Elastic Beanstalk application if prompted

### 🚀 5.2: Create and Deploy the Environment

```bash
eb create nextword-env
```

This will:

-   Create a new EC2 instance with Docker
-   Deploy your app on port 8080

### 🌍 5.3: Open the App in Browser

```bash
eb open
```

---

## 🧪 Step 6: Test the Deployed API

Once deployed, your endpoint will look like:

```
http://nextword-env.eba-xyz.us-east-1.elasticbeanstalk.com/predict
```

Test it using:

```bash
curl -X POST http://<your-url>/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love"}'
```

Expected response:

```json
{
    "input": "I love",
    "next_word": "you" // or something similar
}
```

---

## 🧼 Optional: Cleanup Resources

```bash
eb terminate nextword-env
```

---
