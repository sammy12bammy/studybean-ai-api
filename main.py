from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from transformers import pipeline

app = FastAPI()

class NotesRequest(BaseModel):
    notes: str

class Flashcard(BaseModel):
    question: str
    answer: str

#load model once at startup
generator = pipeline("text2text-generation", "google/flan-t5-small")

def generate_flashcards(notes: str, count: int = 5) -> List[dict]:
    prompt = (
        f"Generate {count} flashcards from the following notes. "
        "Output a JSON array where each item has keys 'question' and 'answer':\n\n"
        f"{notes}"
    )
    output = generator(prompt, max_length=512, num_return_sequences=1)[0]["generated_text"]
    return json.loads(output)


@app.post("/flashcards", response_model=List[Flashcard])
async def create_flashcards(request: NotesRequest):
    cards = generate_flashcards(request.notes)
    return cards

