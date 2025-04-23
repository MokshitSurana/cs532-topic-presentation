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
