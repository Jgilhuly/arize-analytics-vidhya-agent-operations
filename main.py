import os
from getpass import getpass
from typing import List, AsyncGenerator
from langchain_core.messages import HumanMessage
from tqdm import tqdm
from fastapi import FastAPI
from agent import app as agent_app
from phoenix.otel import register
from fastapi import HTTPException
from typing import List
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# Set up environment variables
if os.getenv("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

if os.getenv("PHOENIX_API_KEY") is None:
    os.environ["PHOENIX_API_KEY"] = getpass("Enter your Phoenix API key: ")

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

# Initialize Phoenix tracing
project_name = "agent-operations-course"
tracer_provider = register(
    project_name=project_name,
    auto_instrument=True,
)

def run_single_question(question: str) -> str:
    """Run the agent with a single question"""
    try:
        result = agent_app.invoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)

def run_multiple_questions(questions: List[str]) -> None:
    """Run the agent with multiple questions"""
    for question in tqdm(questions, desc="Processing questions"):
        run_single_question(question)
        
async def stream_agent_response(question: str) -> AsyncGenerator[str, None]:
    """Stream the agent's response for a given question"""
    try:
        for chunk in agent_app.stream(
            {"messages": [HumanMessage(content=question)]}, stream_mode="values"
        ):
            response = chunk["messages"][-1].content
            if response:
                yield f"data: {response}\n\n"
    except Exception as e:
        print(f"Error streaming response for question: {question}")
        print(e)
        yield f"data: Error: {str(e)}\n\n"

app = FastAPI(title="Agent Operations API")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

class QuestionInput(BaseModel):
    question: str

@app.post("/invoke")
async def process_question(input_data: QuestionInput):
    """Process a single question"""
    try:
        # Run the agent directly
        result = run_single_question(input_data.question)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/invoke-streaming")
async def process_question_streaming(input_data: QuestionInput):
    """Process a single question with streaming response"""
    try:
        return StreamingResponse(
            stream_agent_response(input_data.question),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))