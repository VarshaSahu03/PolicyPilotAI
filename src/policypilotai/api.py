
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from policypilotai.graph import stream_user_question


app = FastAPI(title="PolicyPilotAI API")


# CORS configuration for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://policypilotai-ui.onrender.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


# Home route
@app.get("/")
def home():
    return {
        "message": "PolicyPilotAI API is running"
    }


# Chat API with real-time streaming
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    async def generate():

        try:
            async for token in stream_user_question(query):

                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: {\"done\": true}\n\n"

        except Exception as e:
            print("Streaming Error:", repr(e))

            yield f"data: {json.dumps({'error': 'Unable to generate answer'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
