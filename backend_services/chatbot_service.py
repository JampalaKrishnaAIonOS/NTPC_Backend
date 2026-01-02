import pandas as pd
import numpy as np
import json
import os
import duckdb
from typing import Optional, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get project root and data directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"DEBUG: Data Directory: {DATA_DIR}")
print(f"DEBUG: Env Path: {env_path}")
print(f"DEBUG: GROQ_API_KEY present: {bool(GROQ_API_KEY)}")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. Chatbot will use fallback responses.")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("DEBUG: Groq client initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize Groq client: {e}")
        groq_client = None

# Global variables
CONN = None
CURRENT_DATASET = None

def list_available_datasets() -> List[str]:
    """List all available Excel files in the Data folder"""
    try:
        if not os.path.exists(DATA_DIR):
            return []
        # Look for Excel files
        datasets = [f for f in os.listdir(DATA_DIR) if f.endswith((".xlsx", ".xls", ".csv"))]
        return sorted(datasets)
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []

from .data_loader import load_data

def normalize_column(col: str) -> str:
    return (
        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("-", "_")
           .replace("(", "")
           .replace(")", "")
           .replace(".", "")
           .replace("&", "and")
    )

def get_database_connection(dataset_name: Optional[str] = None):
    """Get or create DuckDB connection and load data from data_loader"""
    global CONN, CURRENT_DATASET
    
    if CONN is None:
        CONN = duckdb.connect(":memory:")
    
    # We ignore dataset_name for now as we want to enforce using the centralized data loader
    # This ensures consistency between dashboard and chatbot
    
    if CURRENT_DATASET is None:
        try:
            print("Chatbot loading data via data_loader...")
            data_dict = load_data()
            
            if 'coal_receipt' not in data_dict:
                 raise ValueError("coal_receipt sheet not found in loaded data")
            
            df = data_dict['coal_receipt'].copy()

            # Normalize column names
            df.columns = [normalize_column(c) for c in df.columns]
            
            # Register as 'data' table in DuckDB
            CONN.register("data", df)
            
            # Optional: Register other tables if needed
            if 'coal_stock' in data_dict:
                CONN.register("stock", data_dict['coal_stock'])
            
            CURRENT_DATASET = "Default Dataset"
            
            print(f"Chatbot loaded main dataset ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            print(f"Failed to load dataset for chatbot: {e}")
            raise RuntimeError(f"Chatbot data load failed: {e}")
    
    return CONN

def clean_dataframe_for_json(df):
    """Robust data cleaning for JSON serialization with comprehensive type handling"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        col_dtype = str(df_clean[col].dtype).lower()
        
        # Handle numeric types
        if col_dtype in ['float64', 'float32', 'float16']:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], None)
            df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
            max_float = 1e15
            min_float = -1e15
            df_clean[col] = df_clean[col].where(
                (df_clean[col] <= max_float) & (df_clean[col] >= min_float),
                df_clean[col].astype(str)
            )
        
        # Handle integer types
        elif col_dtype in ['int64', 'int32', 'int16', 'int8']:
            max_int = 2**53 - 1
            min_int = -(2**53 - 1)
            df_clean[col] = df_clean[col].where(
                (df_clean[col] <= max_int) & (df_clean[col] >= min_int),
                df_clean[col].astype(str)
            )
        
        # Handle datetime types
        elif 'datetime' in col_dtype or 'timestamp' in col_dtype:
            try:
                df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df_clean[col] = df_clean[col].astype(str)
        
        # Handle pandas datetime
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            try:
                df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df_clean[col] = df_clean[col].astype(str)
        
        # Handle boolean types
        elif col_dtype in ['bool', 'boolean']:
            df_clean[col] = df_clean[col].astype(bool)
        
        # Handle object/string types
        elif col_dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace(['inf', '-inf', 'nan', 'NaN', 'None', 'null'], None)
            df_clean[col] = df_clean[col].replace('', None)
        
        # Handle categorical types
        elif 'category' in col_dtype:
            df_clean[col] = df_clean[col].astype(str)
        
        # Handle complex types (convert to string)
        elif col_dtype in ['complex64', 'complex128']:
            df_clean[col] = df_clean[col].astype(str)
        
        # Handle any other types by converting to string
        else:
            try:
                df_clean[col] = df_clean[col].astype(str)
            except:
                df_clean[col] = df_clean[col].fillna('').astype(str)
    
    return df_clean

def safe_json_serialize(obj):
    """Safely serialize objects to JSON with comprehensive error handling"""
    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            if abs(int(obj)) > 2**53 - 1:
                return str(obj)
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            if abs(float(obj)) > 1e15:
                return str(obj)
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif hasattr(obj, '__class__') and 'Decimal' in str(obj.__class__):
            if abs(float(obj)) > 1e15:
                return str(obj)
            return float(obj)
        elif hasattr(obj, 'year') and hasattr(obj, 'month'):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(obj, 'dtype') and 'bigint' in str(obj.dtype).lower():
            return str(obj)
        elif isinstance(obj, complex):
            return str(obj)
        elif obj in [float('inf'), float('-inf')] or (isinstance(obj, float) and np.isnan(obj)):
            return None
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_serializer, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        print(f"JSON serialization error: {e}")
        return json.dumps(str(obj), ensure_ascii=False)

def classify_query_type(question: str) -> str:
    """Classify query as conversational or data-related"""
    # If explicitly asking for data, treat as data query
    question_lower = question.lower().strip()
    
    # Specific keywords that STRONGLY suggest a data query
    strong_data_signals = [
        'show me', 'list', 'how many', 'total', 'average', 'maximum', 'minimum', 
        'plot', 'graph', 'chart', 'compare', 'trend', 'cost of', 'vendor'
    ]
    
    # If no Groq client, do simple pattern matching but be conservative
    if not groq_client:
        for signal in strong_data_signals:
            if signal in question_lower:
                return 'data_query'
        return 'conversational'
    
    try:
        prompt = f"""Classify this user query as either 'conversational' or 'data_query'.

Query: "{question}"

Rules:
- 'conversational': Greetings, help requests, general questions (e.g., "Hello", "How are you", "What can you do?")
- 'data_query': Specific data requests, analysis, charts, calculations (e.g., "Show me vendors", "Average cost", "Give me details")

Respond with only: conversational OR data_query"""
        
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        
        classification = message.choices[0].message.content.strip().lower()
        return 'conversational' if 'conversational' in classification else 'data_query'
    except Exception as e:
        print(f"Error in query classification: {e}")
        # Fallback to pattern matching on error
        for signal in strong_data_signals:
            if signal in question_lower:
                return 'data_query'
        return 'conversational'

def generate_sql_from_question(question: str, con: duckdb.DuckDBPyConnection) -> str:
    """Generate SQL query from natural language question"""
    if not groq_client:
        # Fallback - return basic query
        raise RuntimeError("LLM SQL generation failed – check schema or prompt.")
    
    try:
        # Get database schema
        schema_info = con.execute("DESCRIBE data").fetchall()
        columns = [row[0] for row in schema_info]

        schema_text = "Table: data\nColumns:\n"
        schema_text += "\n".join(f"- {c}" for c in columns)
        
        prompt = f"""Convert this question into a SQL query for DuckDB.

DATABASE SCHEMA:
{schema_text}

INSTRUCTIONS:
- Use EXACT table and column names
- Add LIMIT 100 for results
- Use proper GROUP BY when aggregating
- Return ONLY the SQL query, no explanations

Question: "{question}"

SQL Query:"""
        
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        
        sql_query = message.choices[0].message.content.strip()
        
        # Clean up markdown formatting
        if sql_query.startswith('```'):
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        return sql_query
        
    except Exception as e:
        print(f"Error in SQL generation: {e}")
        raise RuntimeError("LLM SQL generation failed – check schema or prompt.")

def create_visualization_from_data(df: pd.DataFrame, question: str) -> Optional[str]:
    """Create visualization from data with chart intent routing"""
    if df.empty or len(df.columns) < 1:
        return None
    
    try:
        question_lower = question.lower()
        
        # Identify column types
        cat_cols = [
            c for c in df.columns
            if df[c].dtype == 'object' or c.endswith('_vendor')
        ]
        num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if not (cat_cols and num_cols):
            return None
        
        x_col = cat_cols[0]
        y_col = num_cols[0]
        df_sorted = df.sort_values(y_col, ascending=False).head(20)
        
        # Chart intent routing
        if "pie" in question_lower:
            # Create pie chart with categorical coloring
            fig = px.pie(
                df_sorted,
                names=x_col,
                values=y_col,
                color=x_col,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title=f"Distribution: {y_col} by {x_col}",
                hole=0.35
            )
            fig.update_layout(height=400)
            
        elif "line" in question_lower or "trend" in question_lower:
            # Create line chart
            fig = px.line(
                df_sorted,
                x=x_col,
                y=y_col,
                title=f"Trend: {y_col} by {x_col}",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400
            )
            
        else:
            # Default: bar chart with categorical coloring
            fig = px.bar(
                df_sorted,
                x=x_col,
                y=y_col,
                color=x_col,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title=f"Analysis: {y_col} by {x_col}"
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False
            )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return None


def generate_conversational_response(question: str) -> str:
    """Generate conversational response"""
    if not groq_client:
        # Fallback responses
        question_lower = question.lower().strip()
        
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! I'm your NTPC Coal Cost Dashboard Assistant. I can help you analyze coal cost data, vendor metrics, and trends. What would you like to know?"
        
        elif any(phrase in question_lower for phrase in ['how are you', 'what can you do', 'help']):
            return """I can help you with:
• Coal cost analysis and metrics
• Vendor comparisons and performance
• Benchmark comparisons
• Cost trends and anomalies
• Detailed data queries

Try asking: Show me average coal cost or Compare vendors"""
        
        else:
            return "I'm here to help with coal cost data analysis. Please ask me a question about your data!"
    
    try:
        prompt = f"""You are a helpful AI assistant for NTPC Coal Cost Dashboard. Respond conversationally.

User query: "{question}"

You can help with coal cost analysis, vendor comparisons, trends, and data insights.

Response:"""
        
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return message.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in conversational response: {e}")
        return "I'm your NTPC Coal Cost Dashboard Assistant. How can I help you today?"

def generate_data_explanation(question: str, sql_query: str, results: List[Dict], df: pd.DataFrame) -> str:
    """Generate explanation for data query results"""
    if not groq_client:
        return f"Found {len(results)} records matching your query."
    
    try:
        sample_data = df.head(3).to_dict('records') if not df.empty else []
        
        data_summary = f"""Query: "{question}"
Results: {len(results)} records
Columns: {list(df.columns) if not df.empty else 'None'}
Sample: {sample_data}"""
        
        prompt = f"""Provide a clear explanation for this data query result.

{data_summary}

Guidelines:
- Explain what the data shows
- Highlight key insights
- Keep it concise and business-friendly

Explanation:"""
        
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return message.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in explanation generation: {e}")
        return f"Found {len(results)} records with relevant data."

def process_chat_query(question: str, dataset_name: str = "") -> Dict[str, Any]:
    """Main function to process chat queries"""
    try:
        # Load the dataset
        con = get_database_connection(dataset_name if dataset_name else None)
        
        # Classify query type
        query_type = classify_query_type(question)
        
        if query_type == 'conversational':
            response_text = generate_conversational_response(question)
            return {
                "success": True,
                "query_type": "conversational",
                "message": {
                    "content": response_text
                },
                "dataset": CURRENT_DATASET
            }
        
        # Handle data queries with intent-based SQL routing
        q = question.lower()
        sql_query = None
        
        # Intent-based SQL map for deterministic queries
        INTENT_SQL_MAP = {
            "vendor": """
                SELECT
                    coal_vendor,
                    SUM(gr_qty) AS total_quantity,
                    SUM(gross_total_coal_z_tot_p_val) AS coal_cost,
                    SUM(gross_total_rail_z_tot_p_val) AS freight_cost
                FROM data
                GROUP BY coal_vendor
                ORDER BY coal_cost DESC
                LIMIT 20
            """,
            "supplier": """
                SELECT
                    coal_vendor,
                    SUM(gr_qty) AS total_quantity,
                    SUM(gross_total_coal_z_tot_p_val) AS coal_cost,
                    SUM(gross_total_rail_z_tot_p_val) AS freight_cost
                FROM data
                GROUP BY coal_vendor
                ORDER BY coal_cost DESC
                LIMIT 20
            """,
            "freight": """
                SELECT
                    coal_vendor,
                    SUM(gross_total_rail_z_tot_p_val) AS freight_cost,
                    SUM(gr_qty) AS total_quantity
                FROM data
                GROUP BY coal_vendor
                ORDER BY freight_cost DESC
                LIMIT 20
            """,
            "quantity": """
                SELECT
                    coal_vendor,
                    SUM(gr_qty) AS total_quantity,
                    SUM(gross_total_coal_z_tot_p_val) AS coal_cost
                FROM data
                GROUP BY coal_vendor
                ORDER BY total_quantity DESC
                LIMIT 20
            """
        }
        
        # Check for intent matches
        for key, sql in INTENT_SQL_MAP.items():
            if key in q:
                sql_query = sql
                break
        
        # Fallback to LLM if no intent matched
        if not sql_query:
            sql_query = generate_sql_from_question(question, con)
        
        # Execute query
        try:
            result_df = con.execute(sql_query).fetchdf()
            result_df = clean_dataframe_for_json(result_df)
            results = result_df.to_dict(orient='records')
            
            # Ensure JSON serialization works
            try:
                json.dumps(results[:5])
            except (TypeError, ValueError) as e:
                print(f"JSON serialization warning: {e}")
                for record in results:
                    for key, value in record.items():
                        try:
                            json.dumps(value)
                        except:
                            record[key] = str(value)
        except Exception as e:
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
                "sql_query": sql_query,
                "dataset": CURRENT_DATASET
            }
        
        # Generate explanation
        print(f"DEBUG: SQL Query: {sql_query}")
        print(f"DEBUG: Result rows: {len(results)}")
        print("DEBUG: Generating explanation...")
        explanation = generate_data_explanation(question, sql_query, results, result_df)
        print(f"DEBUG: Explanation generated: {explanation[:100]}...")
        
        # Create visualization
        print("DEBUG: Creating visualization...")
        visualization = create_visualization_from_data(result_df, question)
        print(f"DEBUG: Visualization created: {bool(visualization)}")
        
        return {
            "success": True,
            "query_type": "data_query",
            "message": {
                "content": explanation,
                "data": {
                    "sql_query": sql_query,
                    "results": results,
                    "visualization": visualization
                }
            },
            "dataset": CURRENT_DATASET
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Server error: {str(e)}"
        }

def get_datasets() -> Dict[str, Any]:
    """Get list of available datasets"""
    try:
        datasets = list_available_datasets()
        return {
            "success": True,
            "datasets": datasets,
            "count": len(datasets),
            "data_dir": DATA_DIR
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_data_schema(dataset_name: str = "") -> Dict[str, Any]:
    """Get data schema information"""
    try:
        con = get_database_connection(dataset_name if dataset_name else None)
        result = con.execute("DESCRIBE data").fetchall()
        columns = [{"name": row[0], "type": row[1]} for row in result]
        return {
            "success": True,
            "columns": columns,
            "dataset": CURRENT_DATASET
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_data_summary(dataset_name: str = "") -> Dict[str, Any]:
    """Get basic data summary"""
    try:
        con = get_database_connection(dataset_name if dataset_name else None)
        
        result = con.execute("""
            SELECT COUNT(*) as total_records FROM data
        """).fetchdf()
        
        summary = result.iloc[0].to_dict()
        summary['dataset'] = CURRENT_DATASET
        
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
