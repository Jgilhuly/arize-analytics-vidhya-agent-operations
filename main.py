import os
from getpass import getpass
from typing import List
from tqdm import tqdm
from agent import run_agent
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

def run_single_question(question: str) -> None:
    """Run the agent with a single question"""
    try:
        run_agent(question)
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)

def run_multiple_questions(questions: List[str]) -> None:
    """Run the agent with multiple questions"""
    for question in tqdm(questions, desc="Processing questions"):
        run_single_question(question)

if __name__ == "__main__":
    # Example questions
    questions = [
        "What was the most popular product SKU?",
        "What was the total revenue across all stores?",
        "Which store had the highest sales volume?",
        "Create a bar chart showing total sales by store",
        "What percentage of items were sold on promotion?",
        "Plot daily sales volume over time",
        "What was the average transaction value?",
        "Create a box plot of transaction values",
        "Which products were frequently purchased together?",
        "Plot a line graph showing the sales trend over time with a 7-day moving average",
    ]

    # Run all questions
    run_multiple_questions(questions) 
    # run_single_question(questions[0])