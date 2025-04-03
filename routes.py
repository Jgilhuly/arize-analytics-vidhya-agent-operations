from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from main import run_single_question, stream_agent_response

router = APIRouter()

class QuestionInput(BaseModel):
    question: str

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.post("/invoke")
async def process_question(input_data: QuestionInput):
    """Process a single question"""
    try:
        # Run the agent directly
        result = run_single_question(input_data.question)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invoke-streaming")
async def process_question_streaming(input_data: QuestionInput):
    """Process a single question with streaming response"""
    try:
        return StreamingResponse(
            stream_agent_response(input_data.question),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))