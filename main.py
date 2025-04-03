import os
from getpass import getpass
from typing import List, AsyncGenerator
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from agent import run_agent, app as agent_app
from phoenix.otel import register

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
        result = run_agent(question)
        return result
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)

async def stream_agent_response(question: str) -> AsyncGenerator[str, None]:
    """Stream the agent's response for a given question"""
    try:
        for chunk in agent_app.stream(
            {"messages": [("human", question)]}, stream_mode="values"
        ):
            response = chunk["messages"][-1].content
            if response:
                yield f"data: {response}\n\n"
    except Exception as e:
        print(f"Error streaming response for question: {question}")
        print(e)
        yield f"data: Error: {str(e)}\n\n"

def run_multiple_questions(questions: List[str]) -> None:
    """Run the agent with multiple questions"""
    for question in tqdm(questions, desc="Processing questions"):
        run_single_question(question)

if __name__ == "__main__":
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    from routes import router

    # Create FastAPI app
    app = FastAPI(title="Agent Operations API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)