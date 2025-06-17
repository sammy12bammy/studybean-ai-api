from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

class NotesRequest(BaseModel):
    notes: str

class Flashcard(BaseModel):
    question: str
    answer: str

# choose your model
MODEL_NAME = "google/flan-t5-small"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load model in 8-bit with automatic device assignment
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# create a text2text pipeline around it
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    num_return_sequences=1
)

def generate_flashcards(notes: str, count: int = 5) -> List[dict]:
    prompt = (
        f"Generate {count} flashcards from the following notes. "
        "Output a JSON array where each item has keys 'question' and 'answer':\n\n"
        f"{notes}"
    )
    raw = generator(prompt)[0]["generated_text"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # if the model output isn’t valid JSON, return an empty list
        return []

@app.post("/flashcards", response_model=List[Flashcard])
async def create_flashcards(request: NotesRequest):
    return generate_flashcards(request.notes)
