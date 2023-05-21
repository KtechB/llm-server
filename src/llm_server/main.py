import io
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
from pydantic import BaseModel, Field

from llm_server.simple_agent import ask_question, create_conversational_chain
from llm_server.whisper import speech_to_text

app = FastAPI()

origins = ["http://localhost:5173"]
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


class WhisperResponse(BaseModel):
    text: str


@app.get("/healthcheck")
def healthcheck():
    return {}


@app.post("/llm")
def run_llm(message: Message) -> LLMResponse:
    answer = ask_question(message.text)
    return LLMResponse(text=answer)


@app.websocket("/chat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # create agent
    chain = create_conversational_chain()
    while True:
        try:
            # Receive and send back the client message
            question = await ws.receive_text()
            answer = chain.predict(input=question)
            resp = LLMResponse(text=answer)
            await ws.send_json(resp.dict())
        except WebSocketDisconnect:
            break
        except Exception as e:
            logging.error(e)
            resp = LLMResponse(
                text="Error happern.",
            )
            await ws.send_json(resp.dict())

@app.post("/whisper/")
async def text_to_speech(file: UploadFile) -> WhisperResponse:
    audio = await file.read()
    buffer = io.BytesIO(audio)
    buffer.name= file.filename 
    text = speech_to_text(buffer)
    return WhisperResponse(text=text)
