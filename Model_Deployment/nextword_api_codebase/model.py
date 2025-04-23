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
