from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_server.simple_agent import ask_question

app = FastAPI()

origins = ["http://localhost:5173", "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str = Field(title="Request message to LLM.", max_length=1000)


class LLMResponse(BaseModel):
    text: str


@app.get("/healthcheck")
def healthcheck():
    return {}


@app.post("/llm")
async def run_llm(message: Message) -> LLMResponse:
    answer = ask_question(message.text)
    return LLMResponse(text=answer)
