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
