import json
from typing import List

import duckdb
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Initialize OpenAI client
model = "gpt-4o-mini"
chat_model = ChatOpenAI(model=model)

# Load the dataset
store_sales_df = pd.read_parquet(
    "https://storage.googleapis.com/arize-phoenix-assets/datasets/unstructured/llm/llama-index/Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

# SQL Generation Prompt
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, columns=columns, table_name=table_name
    )

    response = chat_model.invoke([
        SystemMessage(content=formatted_prompt)
    ])

    return response.content

@tool
def lookup_sales_data(prompt: str) -> str:
    """Look up data from Store Sales Price Elasticity Promotions dataset"""
    try:
        table_name = "sales"

        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM store_sales_df")

        sql_query = generate_sql_query(prompt, store_sales_df.columns, table_name)
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        result = duckdb.sql(sql_query).df()
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"

class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")

@tool
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate Python code to create data visualizations"""
    try:
        prompt = f"""Generate a chart configuration based on this data: {data}
        The goal is to show: {visualization_goal}"""

        response = chat_model.invoke([
            SystemMessage(content=prompt)
        ])

        config = json.loads(response.content)
        
        # Generate Python code for the visualization
        code_prompt = f"""Write python code to create a {config['chart_type']} chart based on the following configuration.
        Only return the code, no other text.
        config: {config}"""

        code_response = chat_model.invoke([
            SystemMessage(content=code_prompt)
        ])

        code = code_response.content
        code = code.replace("```python", "").replace("```", "")
        code = code.strip()

        return code
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

@tool
def run_python_code(code: str) -> str:
    """Run Python code in a restricted environment"""
    try:
        # Create restricted globals/locals dictionaries with plotting libraries
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "round": round,
                "__import__": __import__,
                "json": __import__("json"),
            },
            "plt": __import__("matplotlib.pyplot"),
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "sns": __import__("seaborn"),
        }

        # Execute code in restricted environment
        exec_locals = {}
        exec(code, restricted_globals, exec_locals)

        # Capture any printed output or return the plot
        if "plt" in exec_locals:
            return exec_locals["plt"]

        return "Code executed successfully"
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def analyze_sales_data(prompt: str, data: str) -> str:
    """Analyze sales data to extract insights"""
    try:
        prompt = f"""Analyze the following data: {data}
        Your job is to answer the following question: {prompt}"""

        response = chat_model.invoke([
            SystemMessage(content=prompt)
        ])

        analysis = response.content
        return analysis if analysis else "No analysis could be generated"
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

# Initialize tools
tools = [lookup_sales_data, generate_visualization, run_python_code, analyze_sales_data] 