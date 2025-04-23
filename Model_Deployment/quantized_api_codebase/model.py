from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load StableLM 2 1.6B (Quantized version if available)
model_name = "stabilityai/stablelm-2-1_6b-chat"

token = os.getenv("HF_TOKEN")

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
