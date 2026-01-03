"""
Complete NTPC Coal Cost Dashboard Backend
Exposes all Streamlit functionality as REST APIs
"""

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import json
import math
import threading

import sys
import os
from pathlib import Path

# Fix path to ensure imports work
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import existing services
from backend_services.data_loader import load_data
from backend_services.dashboard import (
    calculate_cost_metrics, 
    detect_anomalies,
    get_overview_payload,
    get_benchmarks_payload,
    build_avoidable_costs_payload,
    get_april_cost_summary,
    get_cost_variants,
    get_coal_vs_freight_pie,
    get_grade_vs_benchmark,
    get_cost_breakdown_variants,
    get_cost_sunburst,
    get_vendor_sunburst,
    get_vendor_kpis
)
# Note: For functions not present in dashboard.py, we'll provide stubs or simplified logic
def get_benchmarks(receipt=None): return BenchmarkData().get_comprehensive_comparison()
def calculate_daily_trends(receipt): return []
def get_suggested_analysis_columns(receipt): return ["GR Qty", "Gross Total (Coal_Z_TOT) P Val"]
# Use implementations from dashboard module where available
build_sunburst_data = get_cost_sunburst
build_vendor_sunburst_data = get_vendor_sunburst
def get_vendor_metrics(receipt): return []
def get_top_vendors(receipt): return []

from backend_services.chatbot_service import (
    process_chat_query, 
    get_datasets, 
    get_data_schema, 
    get_data_summary
)

# Initialize FastAPI
app = FastAPI(
    title="NTPC Coal Cost Dashboard - Complete API",
    description="Complete API for NTPC coal cost analysis with full Streamlit parity",
    version="3.0.0"
)

# Initialize server state for caching
app.state.server_cache = {}
app.state.cache_lock = threading.Lock()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_receipt_data():
    """Get coal receipt data"""
    data = get_all_data()
    if 'coal_receipt' not in data:
        raise KeyError("Coal receipt data not found")
    return data['coal_receipt']


def _make_json_safe(obj):
    """Recursively convert numpy / pandas types to JSON-serializable Python types."""
    # Primitive types
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # numpy scalar
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val

    if isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())

    # pandas types
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return str(obj)
        if isinstance(obj, _pd.Timedelta):
            return str(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]

    # Fallback: try to convert via jsonable_encoder
    try:
        return jsonable_encoder(obj)
    except Exception:
        return str(obj)

def get_all_data():
    """Get all data sheets with in-memory caching for the server lifetime.

    The first call loads from disk via `load_data()`; subsequent calls return the
    cached object until the server process is restarted or the cache is cleared
    via `/api/cache/clear`.
    """
    # Use app.state for better lifecycle management and concurrency
    if '_all_data' in app.state.server_cache:
        return app.state.server_cache['_all_data']

    with app.state.cache_lock:
        if '_all_data' in app.state.server_cache:
            return app.state.server_cache['_all_data']
        data = load_data()
        app.state.server_cache['_all_data'] = data
        return data


# Helper cache accessors for other computed payloads
def _get_cached(key: str):
    return app.state.server_cache.get(key)

def _set_cached(key: str, value: Any):
    app.state.server_cache[key] = value

def _clear_cache(key: Optional[str] = None):
    if key:
        app.state.server_cache.pop(key, None)
    else:
        app.state.server_cache.clear()
    return True

def filter_ntpc_vendors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for NTPC vendors only"""
    if 'Coal Vendor' in df.columns:
        return df[df['Coal Vendor'].str.contains('NTPC', case=False, na=False)]
    return df

def filter_purchases_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out sell transactions"""
    if 'Customer Name' in df.columns:
        return df[df['Customer Name'].isna()]
    return df

# ============================================================================
# BENCHMARK DATA (from dashboard1.py)
# ============================================================================

class BenchmarkData:
    def __init__(self):
        self.industry_standard = 2500  # Open-cast mining
        self.cil_benchmark = 1391     # Coal India Limited average
        self.api2_benchmark = 10120   # Australian international
        
    def get_comprehensive_comparison(self):
        return {
            'industry_standard': self.industry_standard,
            'cil_benchmark': self.cil_benchmark,
            'domestic_average': 2200,
            'australian_intl': self.api2_benchmark
        }
    
    def get_benchmark_table_data(self):
        return [
            {
                'category': 'Industry Standard',
                'geography': 'India - Open-cast mining',
                'benchmark': self.industry_standard,
                'year': '2025',
                'source': 'Times of India - SCCL CMD Interview',
                'url': 'https://timesofindia.com'
            },
            {
                'category': 'National Average',
                'geography': 'India - Coal India Limited',
                'benchmark': self.cil_benchmark,
                'year': '2025',
                'source': 'Government of India Rajya Sabha',
                'url': 'https://rajyasabha.nic.in'
            },
            {
                'category': 'International Benchmark',
                'geography': 'Australia - Mining costs',
                'benchmark': self.api2_benchmark,
                'year': '2023-24',
                'source': 'IEEFA Analysis',
                'url': 'https://ieefa.org'
            }
        ]

# ============================================================================
# NTPC COST ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/ntpc/executive-summary")
async def get_ntpc_executive_summary():
    """Get NTPC executive summary with KPIs"""
    try:
        cache_key = 'overview'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        receipt = filter_ntpc_vendors(receipt)
        
        metrics = calculate_cost_metrics(receipt)
        resp = {
            "success": True,
            "metrics": {
                "total_cost_per_ton": metrics['avg_cost_per_ton'],
                "coal_cost_per_ton": metrics['avg_coal_cost_per_ton'],
                "freight_cost_per_ton": metrics['avg_freight_per_ton'],
                "total_volume": metrics['total_coal_received'],
                "total_coal_cost": metrics['total_coal_cost'],
                "total_freight_cost": metrics['total_freight_cost'],
                "total_cost": metrics['total_cost']
            }
        }
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/ntpc/cost-breakdown")
async def get_ntpc_cost_breakdown():
    """Get NTPC cost breakdown (coal vs freight)"""
    try:
        cache_key = 'ntpc_cost_breakdown'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        receipt = filter_ntpc_vendors(receipt)
        
        metrics = calculate_cost_metrics(receipt)
        
        total_cost = metrics['total_cost']
        coal_cost = metrics['total_coal_cost']
        freight_cost = metrics['total_freight_cost']
        resp = {
            "success": True,
            "breakdown": {
                "coal_cost": float(coal_cost),
                "freight_cost": float(freight_cost),
                "total_cost": float(total_cost),
                "coal_percentage": float((coal_cost / total_cost * 100) if total_cost > 0 else 0),
                "freight_percentage": float((freight_cost / total_cost * 100) if total_cost > 0 else 0)
            }
        }
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/ntpc/benchmark-comparison")
async def get_ntpc_benchmark_comparison():
    """Get NTPC benchmark comparison data"""
    try:
        cache_key = 'ntpc_benchmark_comparison'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        receipt = filter_ntpc_vendors(receipt)
        
        metrics = calculate_cost_metrics(receipt)
        coal_cost_per_ton = metrics['avg_coal_cost_per_ton']
        
        benchmarks = BenchmarkData()
        industry_benchmark = benchmarks.industry_standard
        cil_benchmark = benchmarks.cil_benchmark
        australian_benchmark = benchmarks.api2_benchmark
        resp = {
            "success": True,
            "ntpc_coal_cost": float(coal_cost_per_ton),
            "benchmarks": {
                "industry_standard": {
                    "value": float(industry_benchmark),
                    "delta": float(coal_cost_per_ton - industry_benchmark),
                    "delta_percentage": float((coal_cost_per_ton - industry_benchmark) / industry_benchmark * 100) if industry_benchmark > 0 else 0
                },
                "cil_benchmark": {
                    "value": float(cil_benchmark),
                    "delta": float(coal_cost_per_ton - cil_benchmark),
                    "delta_percentage": float((coal_cost_per_ton - cil_benchmark) / cil_benchmark * 100) if cil_benchmark > 0 else 0
                },
                "australian_benchmark": {
                    "value": float(australian_benchmark),
                    "delta": float(coal_cost_per_ton - australian_benchmark),
                    "delta_percentage": float((coal_cost_per_ton - australian_benchmark) / australian_benchmark * 100) if australian_benchmark > 0 else 0
                }
            }
        }
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/ntpc/daily-trends")
async def get_ntpc_daily_trends():
    """Get NTPC daily cost trends"""
    try:
        cache_key = 'ntpc_daily_trends'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        receipt = filter_ntpc_vendors(receipt)
        
        if 'Entry Dt' not in receipt.columns:
            return JSONResponse({"success": False, "error": "Entry Dt column not found"}, status_code=400)
        
        daily_costs = receipt.groupby('Entry Dt').agg({
            'Gross Total (Coal_Z_TOT) P Val': 'sum',
            'Gross Total (Rail_Z_TOT) P Val': 'sum',
            'GR Qty': 'sum'
        }).reset_index()
        
        daily_costs['Total Cost'] = (
            daily_costs['Gross Total (Coal_Z_TOT) P Val'] + 
            daily_costs['Gross Total (Rail_Z_TOT) P Val']
        )
        daily_costs['Cost per Ton'] = daily_costs['Total Cost'] / daily_costs['GR Qty']
        daily_costs['Entry Dt'] = daily_costs['Entry Dt'].astype(str)
        resp = {"success": True, "trends": daily_costs.to_dict(orient="records")}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/ntpc/comprehensive-benchmarks")
async def get_comprehensive_benchmarks():
    """Get comprehensive benchmark comparison"""
    try:
        cache_key = 'comprehensive_benchmarks'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        receipt = filter_ntpc_vendors(receipt)
        
        metrics = calculate_cost_metrics(receipt)
        
        benchmarks = BenchmarkData()
        comprehensive_data = benchmarks.get_comprehensive_comparison()
        comprehensive_data['ntpc_current'] = metrics['avg_cost_per_ton']
        resp = {"success": True, "comparison": comprehensive_data}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/ntpc/benchmark-references")
async def get_benchmark_references():
    """Get benchmark reference table data"""
    try:
        cache_key = 'benchmark_references'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        benchmarks = BenchmarkData()
        table_data = benchmarks.get_benchmark_table_data()
        resp = {"success": True, "references": table_data}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# VENDOR ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/vendor/comparison")
async def get_vendor_comparison():
    """Get vendor comparison metrics"""
    try:
        cache_key = 'vendor_comparison'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        resp = {
            "success": True,
            "vendor_metrics": get_vendor_metrics(receipt),
            "top_vendors": get_top_vendors(receipt)
        }
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/vendor/detailed-analysis")
async def get_vendor_detailed_analysis():
    """Get detailed vendor analysis"""
    try:
        cache_key = 'vendor_detailed_analysis'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        receipt = filter_purchases_only(receipt)
        
        # Calculate vendor-wise metrics
        vendor_analysis = receipt.groupby('Coal Vendor').agg({
            'GR Qty': 'sum',
            'Gross Total (Coal_Z_TOT) P Val': 'sum',
            'Gross Total (Rail_Z_TOT) P Val': 'sum'
        }).reset_index()
        
        vendor_analysis['Total Cost'] = (
            vendor_analysis['Gross Total (Coal_Z_TOT) P Val'] + 
            vendor_analysis['Gross Total (Rail_Z_TOT) P Val']
        )
        vendor_analysis['Cost per Ton'] = vendor_analysis['Total Cost'] / vendor_analysis['GR Qty']
        vendor_analysis['Market Share'] = (vendor_analysis['GR Qty'] / vendor_analysis['GR Qty'].sum() * 100)
        vendor_analysis['Vendor Type'] = vendor_analysis['Coal Vendor'].apply(
            lambda x: 'NTPC' if 'NTPC' in str(x).upper() else 'Others'
        )
        resp = {"success": True, "vendors": vendor_analysis.to_dict(orient="records")}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# SUNBURST ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/sunburst/cost-hierarchy")
async def get_cost_sunburst():
    """Get cost hierarchy sunburst data"""
    try:
        cache_key = 'sunburst_cost'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        payload = build_sunburst_data(receipt)
        resp = {"success": True, "data": payload}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dashboard/sunburst/vendor")
async def get_vendor_sunburst():
    """Get vendor sunburst hierarchy"""
    try:
        cache_key = 'sunburst_vendor'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        payload = build_vendor_sunburst_data(receipt)
        resp = {"success": True, "data": payload}
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# ANOMALY DETECTION ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/anomalies/suggested-columns")
async def get_anomaly_suggested_columns():
    """Get suggested columns for anomaly analysis"""
    try:
        receipt = get_receipt_data()
        columns = get_suggested_analysis_columns(receipt)
        
        return JSONResponse({
            "success": True,
            "columns": columns
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/dashboard/anomalies/detect")
async def detect_data_anomalies(contamination: Optional[float] = Form(0.05)):
    """Detect anomalies in data"""
    try:
        cache_key = f"anomalies_{float(contamination)}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        columns = get_suggested_analysis_columns(receipt)
        anomalies_df = detect_anomalies(receipt, columns_to_analyze=columns, contamination=contamination)

        # detect_anomalies returns a DataFrame; convert to records and standardize keys
        if anomalies_df is None or (hasattr(anomalies_df, 'empty') and anomalies_df.empty):
            records = []
        else:
            records = anomalies_df.copy()
            # Ensure boolean flag is named 'is_anomaly' for frontend
            if 'anomaly' in records.columns:
                records = records.rename(columns={'anomaly': 'is_anomaly'})
            # Keep anomaly_score if present
            # Convert timestamp/index types to strings for JSON safety
            for col in records.select_dtypes(include=['datetime']).columns:
                records[col] = records[col].astype(str)
            records = records.to_dict(orient='records')

        anomaly_count = sum(1 for r in records if r.get('is_anomaly') is True)
        total_count = len(records)
        anomaly_percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0

        payload = {
            "success": True,
            "anomalies": records,
            "analysis_columns": columns,
            "statistics": {
                "anomaly_count": anomaly_count,
                "total_count": total_count,
                "anomaly_percentage": float(anomaly_percentage)
            }
        }
        # Cache anomalies result for this contamination until server restart/clear
        _set_cached(cache_key, payload)
        safe = _make_json_safe(payload)
        return JSONResponse(safe)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# OPERATIONS ENDPOINTS
# ============================================================================

@app.get("/api/dashboard/operations/overview")
async def get_operations_overview():
    """Get operations overview"""
    try:
        cache_key = 'operations_overview'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        receipt = get_receipt_data()
        metrics = calculate_cost_metrics(receipt)
        resp = {
            "success": True,
            "total_coal_received": metrics['total_coal_received'],
            "total_cost": metrics['total_cost'],
            "average_cost_per_ton": metrics['avg_cost_per_ton'],
            "record_count": len(receipt)
        }
        _set_cached(cache_key, resp)
        return JSONResponse(_make_json_safe(resp))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# DATA UPLOAD & VALIDATION ENDPOINTS
# ============================================================================

@app.post("/api/data/validate")
async def validate_data_upload(file: UploadFile = File(...)):
    """Validate uploaded Excel file"""
    try:
        # Read Excel file
        contents = await file.read()
        excel_file = pd.ExcelFile(contents)
        
        # Required sheets
        required_sheets = [
            'Daywise_CoalStock',
            'Daywise_CoalConsumption',
            'Daywise_CoalReceipt_C&FCost',
            'Daywise_GCVData',
            'Siding Details',
            'Mine Code'
        ]
        
        # Check for required sheets
        missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
        
        if missing_sheets:
            return JSONResponse({
                "success": False,
                "error": f"Missing required sheets: {', '.join(missing_sheets)}",
                "available_sheets": excel_file.sheet_names
            }, status_code=400)
        
        # Validate coal receipt sheet columns
        coal_receipt = pd.read_excel(contents, sheet_name='Daywise_CoalReceipt_C&FCost')
        required_columns = [
            'Coal Vendor',
            'GR Qty',
            'Gross Total (Coal_Z_TOT) P Val',
            'Gross Total (Rail_Z_TOT) P Val',
            'Entry Dt'
        ]
        
        missing_columns = [col for col in required_columns if col not in coal_receipt.columns]
        
        if missing_columns:
            return JSONResponse({
                "success": False,
                "error": f"Missing required columns in CoalReceipt sheet: {', '.join(missing_columns)}",
                "available_columns": list(coal_receipt.columns)
            }, status_code=400)
        
        return JSONResponse({
            "success": True,
            "message": "File validation successful",
            "sheets": excel_file.sheet_names,
            "record_count": len(coal_receipt)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Validation error: {str(e)}"
        }, status_code=500)

# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@app.post("/api/chat")
async def chat_with_assistant(question: str = Form(...), dataset: str = Form("")):
    """Chatbot endpoint"""
    return JSONResponse(process_chat_query(question, dataset if dataset else ""))

@app.get("/api/datasets")
async def list_datasets():
    return JSONResponse(get_datasets())

@app.get("/api/data/schema")
async def show_schema(dataset: str = ""):
    return JSONResponse(get_data_schema(dataset if dataset else ""))

@app.get("/api/data/summary")
async def show_summary(dataset: str = ""):
    return JSONResponse(get_data_summary(dataset if dataset else ""))

# ============================================================================
# UNIFIED DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/api/overview")
async def api_overview():
    """Get unified dashboard overview"""
    try:
        return JSONResponse(get_overview_payload())
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/benchmarks")
async def api_benchmarks():
    """Get unified dashboard benchmarks"""
    try:
        return JSONResponse(get_benchmarks_payload())
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/avoidable-costs")
async def api_avoidable_costs():
    """Get unified dashboard avoidable costs"""
    try:
        cache_key = 'avoidable_costs'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        payload = build_avoidable_costs_payload()
        _set_cached(cache_key, payload)
        safe = _make_json_safe(payload)
        return JSONResponse(safe)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/cache/clear")
async def clear_cache_endpoint(key: Optional[str] = None):
    """Clear server-side cache. If `key` provided, clears that entry,
    otherwise clears the entire cache."""
    try:
        success = _clear_cache(key)
        if not success:
            return JSONResponse({"success": False, "error": "Cache not initialized"}, status_code=400)
        return JSONResponse({"success": True, "cleared_key": key})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# APRIL ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/april/summary")
async def api_april_summary():
    try:
        return JSONResponse(jsonable_encoder(get_april_cost_summary()))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/april/cost-variants")
async def api_april_cost_variants():
    try:
        return JSONResponse(jsonable_encoder(get_cost_variants()))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/april/coal-vs-freight")
async def api_april_coal_freight():
    try:
        return JSONResponse(jsonable_encoder(get_coal_vs_freight_pie()))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/april/grade-benchmark")
async def api_april_grade_benchmark():
    try:
        cache_key = 'april_grade_benchmark'
        cached = _get_cached(cache_key)
        if cached is not None:
            return JSONResponse(_make_json_safe(cached))

        payload = get_grade_vs_benchmark()
        _set_cached(cache_key, payload)
        safe = _make_json_safe(payload)
        return JSONResponse(safe)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/april/cost-breakdown")
async def api_april_cost_breakdown():
    try:
        return JSONResponse(jsonable_encoder(get_cost_breakdown_variants()))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/vendors/kpis")
async def api_vendor_kpis():
    try:
        return JSONResponse(get_vendor_kpis())
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ============================================================================
# HEALTH & INFO
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "NTPC Coal Cost Dashboard - Complete API",
        "version": "3.0.0",
        "status": "active",
        "endpoints": {
            "ntpc_analysis": "/api/dashboard/ntpc/*",
            "vendor_analysis": "/api/dashboard/vendor/*",
            "sunburst": "/api/dashboard/sunburst/*",
            "anomalies": "/api/dashboard/anomalies/*",
            "operations": "/api/dashboard/operations/*",
            "data": "/api/data/*",
            "chat": "/api/chat"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "NTPC Backend Complete"}

@app.get("/api/endpoints")
async def list_endpoints():
    """List all available endpoints"""
    return {
        "ntpc_cost_analysis": [
            "/api/dashboard/ntpc/executive-summary",
            "/api/dashboard/ntpc/cost-breakdown",
            "/api/dashboard/ntpc/benchmark-comparison",
            "/api/dashboard/ntpc/daily-trends",
            "/api/dashboard/ntpc/comprehensive-benchmarks",
            "/api/dashboard/ntpc/benchmark-references"
        ],
        "vendor_analysis": [
            "/api/dashboard/vendor/comparison",
            "/api/dashboard/vendor/detailed-analysis"
        ],
        "sunburst": [
            "/api/dashboard/sunburst/cost-hierarchy",
            "/api/dashboard/sunburst/vendor"
        ],
        "anomalies": [
            "/api/dashboard/anomalies/suggested-columns",
            "/api/dashboard/anomalies/detect"
        ],
        "operations": [
            "/api/dashboard/operations/overview"
        ],
        "data": [
            "/api/data/validate",
            "/api/data/schema",
            "/api/data/summary"
        ],
        "chatbot": [
            "/api/chat",
            "/api/datasets"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

