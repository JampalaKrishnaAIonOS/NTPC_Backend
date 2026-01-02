"""
Coal Cost Optimization Backend

Unifies logic from:
- coal_cost_dashboard.py
- avoidable_costs_streamlit.py
- benchmark_loader.py
- benchmarks.yaml / CSV

Exposes pure functions that a web framework (FastAPI/Flask) can serve as JSON.
"""

import os
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from .data_loader import DATA_ROOT
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
#   Benchmark Loader (from benchmark_loader.py)
# -----------------------------

class BenchmarkLoader:
    """Load and manage coal cost benchmarks from CSV file"""

    def __init__(self, csv_file: str = "references/coal_cost_benchmarks_with_urls.csv"):
        self.csv_file = csv_file
        self.benchmarks_df = self._load_benchmarks()
        self.benchmarks = self._process_benchmarks()

    def _load_benchmarks(self) -> pd.DataFrame:
        try:
            # Use DATA_ROOT for benchmarks if they are in the Data folder, 
            # otherwise keep relative if they are in references/
            path = DATA_ROOT / "references" / os.path.basename(self.csv_file) if not os.path.isabs(self.csv_file) else self.csv_file
            if not os.path.exists(path):
                # Fallback to local if not in Data/references
                path = self.csv_file
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            print(f"Warning: {self.csv_file} not found. Using default benchmarks.")
            return self._get_default_dataframe()
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return self._get_default_dataframe()

    def _get_default_dataframe(self) -> pd.DataFrame:
        default_data = {
            "Category": [
                "Industry-standard benchmark",
                "National average benchmark",
                "International benchmark",
            ],
            "Geography / Scope": ["India", "India", "International"],
            "Benchmark (unit)": ["₹2,500/t", "₹2,600/t", "₹6,771/t"],
            "Year / Period": ["2024", "2024", "2024"],
            "Source / Reference": ["Default", "Default", "Default"],
            "URL": ["", "", ""],
        }
        return pd.DataFrame(default_data)

    def _process_benchmarks(self) -> Dict[str, Any]:
        benchmarks: Dict[str, Any] = {}
        for _, row in self.benchmarks_df.iterrows():
            category = row["Category"]
            benchmark_str = str(row["Benchmark (unit)"])
            cost_value = self._extract_cost_value(benchmark_str)
            if cost_value is not None:
                if category not in benchmarks:
                    benchmarks[category] = {}
                key = row["Geography / Scope"].lower().replace(" ", "_").replace("/", "_")
                benchmarks[category][key] = {
                    "cost_per_ton": cost_value,
                    "source": row["Source / Reference"],
                    "url": row["URL"],
                    "year": row["Year / Period"],
                    "description": f"{row['Geography / Scope']} - {row['Source / Reference']}",
                }
        return benchmarks

    def _extract_cost_value(self, benchmark_str: str) -> Optional[float]:
        import re
        USD_TO_INR = 83.0
        AUD_TO_INR = 55.0

        aud_match = re.search(r"A\$([\d,]+\.?\d*)", benchmark_str)
        if aud_match:
            value = float(aud_match.group(1).replace(",", ""))
            return value * AUD_TO_INR

        usd_match = re.search(r"\$([\d,]+\.?\d*)", benchmark_str)
        if usd_match:
            value = float(usd_match.group(1).replace(",", ""))
            return value * USD_TO_INR

        inr_match = re.search(r"₹([\d,]+\.?\d*)", benchmark_str)
        if inr_match:
            value = float(inr_match.group(1).replace(",", ""))
            return value

        number_match = re.search(r"([\d,]+\.?\d*)", benchmark_str)
        if number_match:
            value = float(number_match.group(1).replace(",", ""))
            return value

        return None

    # ---- Public benchmark methods (unchanged logic) ----

    def get_major_players(self) -> Dict[str, Dict[str, Any]]:
        return self.benchmarks.get("Industry-standard benchmark", {})

    def get_cil_benchmark(self) -> float:
        cil_data = self.benchmarks.get(
            "National average benchmark – cost of production", {}
        ).get("india_–_coal_india_ltd_(cil)", {})
        return cil_data.get("cost_per_ton", 1391.0)

    def get_sccl_benchmark(self) -> float:
        sccl_data = self.benchmarks.get(
            "National average benchmark – cost of production", {}
        ).get("india_–_singareni_collieries_(sccl)", {})
        return sccl_data.get("cost_per_ton", 2877.0)

    def get_commercial_miners_benchmark(self) -> float:
        industry_data = self.benchmarks.get(
            "Industry-standard benchmark (method split)", {}
        ).get("india_(sccl_cmd_disclosure)", {})
        return industry_data.get("cost_per_ton", 2500.0)

    def get_industry_standards(self) -> Dict[str, Dict[str, Any]]:
        return self.benchmarks.get("Industry-standard benchmark (method split)", {})

    def get_domestic_coal_average(self) -> float:
        return self.get_cil_benchmark()

    def get_imported_coal_average(self) -> float:
        return self.get_international_average()

    def get_delivered_price_benchmark(self) -> float:
        us_data = self.benchmarks.get(
            "International benchmark – proxy (delivered less transport)", {}
        ).get("united_states_(power_sector_deliveries)", {})
        return us_data.get("cost_per_ton", 2075.0)

    def get_national_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        return self.benchmarks.get("National average benchmark – cost of production", {})

    def get_government_benchmark(self) -> float:
        return self.get_cil_benchmark()

    def get_regulatory_benchmark(self) -> float:
        return (self.get_cil_benchmark() + self.get_sccl_benchmark()) / 2.0

    def get_national_average(self) -> float:
        return (self.get_cil_benchmark() + self.get_sccl_benchmark()) / 2.0

    def get_international_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        return self.benchmarks.get("International benchmark – unit mining cost", {})

    def get_api2_benchmark(self) -> float:
        aus_data = self.benchmarks.get(
            "International benchmark – unit mining cost", {}
        ).get("australia_(coal_producers;_company-reported_unit_costs)", {})
        return aus_data.get("cost_per_ton", 10120.0)

    def get_newcastle_benchmark(self) -> float:
        return self.get_api2_benchmark()

    def get_indonesian_benchmark(self) -> float:
        us_data = self.benchmarks.get(
            "International benchmark – proxy (mine FOB sales price)", {}
        ).get("united_states_(all_coal)", {})
        return us_data.get("cost_per_ton", 4485.0)

    def get_international_average(self) -> float:
        aus_cost = self.get_api2_benchmark()
        us_cost = self.get_indonesian_benchmark()
        return (aus_cost + us_cost) / 2.0

    # Convenience: formatted table version for frontend
    def get_benchmark_table(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in self.benchmarks_df.iterrows():
            cost_value = self._extract_cost_value(str(row["Benchmark (unit)"]))
            if cost_value is None:
                continue
            rows.append(
                {
                    "category": row["Category"],
                    "geography_scope": row["Geography / Scope"],
                    "benchmark_inr_per_ton": cost_value,
                    "year_period": row["Year / Period"],
                    "source": row["Source / Reference"],
                    "url": row["URL"],
                }
            )
        return rows


benchmark_loader = BenchmarkLoader()

# -----------------------------
#   Core coal cost data loader (from coal_cost_dashboard.py)
# -----------------------------

def load_data(
    file_path: str = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    if file_path is None:
        file_path = DATA_ROOT / "Dadri_Data_April.xlsx"
    try:
        coal_stock = pd.read_excel(file_path, sheet_name="Daywise_CoalStock")
        coal_consumption = pd.read_excel(file_path, sheet_name="Daywise_CoalConsumption")
        coal_receipt = pd.read_excel(
            file_path, sheet_name="Daywise_CoalReceipt_C&FCost"
        )
        gcv_data = pd.read_excel(file_path, sheet_name="Daywise_GCVData")
        siding_details = pd.read_excel(file_path, sheet_name="Siding Details")
        mine_code = pd.read_excel(file_path, sheet_name="Mine Code")

        # Normalize column names (strip whitespace) for all loaded sheets
        for df in (coal_stock, coal_consumption, coal_receipt, gcv_data, siding_details, mine_code):
            df.columns = df.columns.str.strip()

        return {
            "coal_stock": coal_stock,
            "coal_consumption": coal_consumption,
            "coal_receipt": coal_receipt,
            "gcv_data": gcv_data,
            "siding_details": siding_details,
            "mine_code": mine_code,
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# -----------------------------
#   Cost metrics & anomaly detection (coal_cost_dashboard.py)
# -----------------------------

def calculate_cost_metrics(
    data: Dict[str, pd.DataFrame],
    vendor_filter: Optional[str] = None,
) -> Dict[str, Any]:
    if data is None:
        return {}

    coal_receipt = data["coal_receipt"].copy()

    if "Customer Name" in coal_receipt.columns:
        coal_receipt = coal_receipt[coal_receipt["Customer Name"].isna()]

    if vendor_filter and "Coal Vendor" in coal_receipt.columns:
        if vendor_filter == "NTPC":
            coal_receipt = coal_receipt[
                coal_receipt["Coal Vendor"].str.contains(
                    "NTPC", case=False, na=False
                )
            ]
        else:
            coal_receipt = coal_receipt[
                coal_receipt["Coal Vendor"] == vendor_filter
            ]

    cost_metrics: Dict[str, Any] = {}

    if (
        "GR Qty" in coal_receipt.columns
        and "Gross Total (Coal_Z_TOT) P Val" in coal_receipt.columns
    ):
        valid_data = coal_receipt[
            (coal_receipt["GR Qty"] > 0)
            & (coal_receipt["Gross Total (Coal_Z_TOT) P Val"] > 0)
        ]
        if len(valid_data) > 0:
            total_qty = float(valid_data["GR Qty"].sum())
            total_cost = float(valid_data["Gross Total (Coal_Z_TOT) P Val"].sum())
            cost_metrics["avg_cost_per_ton"] = total_cost / total_qty
            cost_metrics["total_coal_received"] = total_qty
            cost_metrics["total_coal_cost"] = total_cost

    if "Gross Total (Rail_Z_TOT) P Val" in coal_receipt.columns:
        total_freight = float(
            coal_receipt["Gross Total (Rail_Z_TOT) P Val"].sum()
        )
        cost_metrics["total_freight_cost"] = total_freight
        if cost_metrics.get("total_coal_received", 0) > 0:
            cost_metrics["avg_freight_per_ton"] = (
                total_freight / cost_metrics["total_coal_received"]
            )

    return cost_metrics


def detect_anomalies(
    df: pd.DataFrame,
    columns_to_analyze: Optional[List[str]] = None,
    contamination: float = 0.1
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if not columns_to_analyze:
        # Default to common numeric columns if none provided
        columns_to_analyze = [
            "GR Qty", 
            "Gross Total (Coal_Z_TOT) P Val", 
            "Gross Total (Rail_Z_TOT) P Val"
        ]
        # Only keep columns that actually exist
        columns_to_analyze = [c for c in columns_to_analyze if c in df.columns]
        
    if not columns_to_analyze:
        return pd.DataFrame()

    numeric_data = df[columns_to_analyze].select_dtypes(include=[np.number])
    # If there are no numeric columns or no rows, return empty
    if numeric_data.shape[1] == 0 or numeric_data.shape[0] == 0:
        return pd.DataFrame()
    numeric_data = numeric_data.fillna(numeric_data.median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_data)

    result = df.copy()
    result["anomaly"] = anomaly_labels == -1
    result["anomaly_score"] = iso_forest.decision_function(scaled_data)
    return result


# -----------------------------
#   Avoidable costs logic (from avoidable_costs_streamlit.py)
# -----------------------------

def _clean_component_name(column: str) -> str:
    return (
        column.replace("(Coal_", " (")
        .replace(") P Val", ")")
        .replace(") C Val", ")")
    )


def load_avoidable_costs_data(
    type_of_costs_path: str = None,
    dadri_data_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if type_of_costs_path is None:
        type_of_costs_path = DATA_ROOT / "type_of_costs.xlsx"
    if dadri_data_path is None:
        dadri_data_path = DATA_ROOT / "Dadri_Data_April.xlsx"
    
    print(f"DEBUG: type_of_costs_path = {type_of_costs_path}")
    print(f"DEBUG: dadri_data_path = {dadri_data_path}")
    
    try:
        if not os.path.exists(type_of_costs_path):
            raise FileNotFoundError(f"Missing file: {type_of_costs_path}")

        xl = pd.ExcelFile(type_of_costs_path)

        # Prefer exact sheet name, otherwise try fuzzy/first-sheet fallback
        preferred_names = ["Avoidable Cost Columns", "Avoidable_Cost_Columns", "Avoidable Cost"]
        sheet_name = None
        for name in preferred_names:
            if name in xl.sheet_names:
                sheet_name = name
                break
        if sheet_name is None:
            # Try partial match
            for name in xl.sheet_names:
                if "avoidable" in name.lower():
                    sheet_name = name
                    break
        if sheet_name is None:
            # Last resort: pick first sheet and warn
            sheet_name = xl.sheet_names[0]
            print(f"WARNING: Using first sheet '{sheet_name}' for avoidable costs (no obvious 'avoidable' sheet found). Available: {xl.sheet_names}")

        avoidable_costs = pd.read_excel(type_of_costs_path, sheet_name=sheet_name)

        # Normalize column names to be robust to small differences
        avoidable_costs.columns = [str(c).strip() for c in avoidable_costs.columns]

        # Accept common variants for required columns
        col_map = {}
        lowered = {c.lower(): c for c in avoidable_costs.columns}
        if "category" in lowered:
            col_map["Category"] = lowered["category"]
        elif "cat" in lowered:
            col_map["Category"] = lowered["cat"]

        # Column name variants: 'column', 'column name', 'column_name'
        for key in ("column", "column name", "column_name", "columnname"):
            if key in lowered:
                col_map["Column"] = lowered[key]
                break

        # If any required column missing, attempt fuzzy match by substring
        if "Category" not in col_map:
            for c in avoidable_costs.columns:
                if "category" in str(c).lower():
                    col_map["Category"] = c
                    break
        if "Column" not in col_map:
            for c in avoidable_costs.columns:
                if "column" in str(c).lower() or "col" == str(c).strip().lower():
                    col_map["Column"] = c
                    break

        required_cols = ["Category", "Column"]
        missing_cols = [r for r in required_cols if r not in col_map]
        if missing_cols:
            print(f"ERROR: Missing/ambiguous columns in avoidable costs sheet: {missing_cols}")
            print(f"Available columns: {list(avoidable_costs.columns)}")
            raise ValueError(f"Missing columns in avoidable costs sheet: {missing_cols}")

        # Rename to canonical column names for downstream logic
        avoidable_costs = avoidable_costs.rename(columns={col_map["Category"]: "Category", col_map["Column"]: "Column"})

        coal_receipt = pd.read_excel(dadri_data_path, sheet_name="Daywise_CoalReceipt_C&FCost")
        return avoidable_costs, coal_receipt
    except Exception as e:
        print(f"FATAL ERROR in load_avoidable_costs_data: {e}")
        raise


def process_avoidable_costs_data(
    avoidable_costs: pd.DataFrame,
    coal_receipt: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    category_columns: Dict[str, List[str]] = {}
    for _, row in avoidable_costs.iterrows():
        category = row["Category"]
        column = row["Column"]
        if column in coal_receipt.columns:
            category_columns.setdefault(category, []).append(column)

    results: Dict[str, Dict[str, Any]] = {}
    for category, columns in category_columns.items():
        category_data = coal_receipt[["Entry Dt"] + columns].copy()
        category_data[columns] = category_data[columns].fillna(0)
        daily_totals = (
            category_data.groupby("Entry Dt")[columns].sum().reset_index()
        )
        daily_totals["Total"] = daily_totals[columns].sum(axis=1)

        results[category] = {
            "daily_data": daily_totals,
            "columns": columns,
            "total_cost": float(daily_totals["Total"].sum()),
            "avg_daily_cost": float(daily_totals["Total"].mean()),
            "max_daily_cost": float(daily_totals["Total"].max()),
            "min_daily_cost": float(daily_totals["Total"].min()),
        }
    return results


def create_category_stack_chart_figure(
    category: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    daily_data = data["daily_data"].copy()
    columns = data["columns"]
    daily_data["Date"] = pd.to_datetime(daily_data["Entry Dt"])
    daily_data = daily_data.sort_values("Date")

    fig = go.Figure()
    colors = px.colors.qualitative.Set3[: len(columns)]

    for i, column in enumerate(columns):
        fig.add_trace(
            go.Bar(
                name=_clean_component_name(column),
                x=daily_data["Date"],
                y=daily_data[column],
                marker_color=colors[i % len(colors)],
                hovertemplate=(
                    f"{_clean_component_name(column)}<br>"
                    "Date: %{x}<br>"
                    "Value: ₹%{y:,.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"📊 {category} - Daily Cost Breakdown",
        xaxis_title="Date",
        yaxis_title="Cost (₹)",
        barmode="stack",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    return fig.to_dict()


def create_category_summary_chart_figure(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    categories = list(results.keys())
    total_costs = [results[cat]["total_cost"] for cat in categories]
    avg_costs = [results[cat]["avg_daily_cost"] for cat in categories]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Total Avoidable Costs by Category",
            "Average Daily Costs by Category",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=total_costs,
            name="Total Cost",
            marker_color="#e74c3c",
            hovertemplate="%{x}<br>Total Cost: ₹%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=avg_costs,
            name="Avg Daily Cost",
            marker_color="#3498db",
            hovertemplate="%{x}<br>Avg Daily Cost: ₹%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="📈 Avoidable Costs Summary",
        title_x=0.5,
    )
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹", row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹", row=1, col=2)
    return fig.to_dict()


def build_avoidable_costs_payload() -> Dict[str, Any]:
    avoidable_costs, coal_receipt = load_avoidable_costs_data()
    results = process_avoidable_costs_data(avoidable_costs, coal_receipt)

    total_avoidable_cost = float(
        sum(results[cat]["total_cost"] for cat in results.keys())
    )
    total_avg_daily = float(
        sum(results[cat]["avg_daily_cost"] for cat in results.keys())
    )
    summary_chart = create_category_summary_chart_figure(results)

    categories_payload = {}
    for category, data in results.items():
        df = data["daily_data"]
        component_rows = []
        for column in data["columns"]:
            total_value = float(df[column].sum())
            avg_value = float(df[column].mean())
            max_value = float(df[column].max())
            share = (
                (total_value / data["total_cost"] * 100)
                if data["total_cost"] > 0
                else 0.0
            )
            component_rows.append(
                {
                    "component": _clean_component_name(column),
                    "total_cost": total_value,
                    "avg_daily": avg_value,
                    "max_daily": max_value,
                    "share_percent": share,
                }
            )
        component_rows = sorted(
            component_rows, key=lambda r: r["total_cost"], reverse=True
        )

        categories_payload[category] = {
            "metrics": {
                "total_cost": data["total_cost"],
                "avg_daily_cost": data["avg_daily_cost"],
                "max_daily_cost": data["max_daily_cost"],
                "min_daily_cost": data["min_daily_cost"],
                "components": len(data["columns"]),
            },
            "stacked_chart": create_category_stack_chart_figure(category, data),
            "component_table": {"category": category, "rows": component_rows},
        }

    comparison_rows = []
    for category, data in results.items():
        share = (
            data["total_cost"] / total_avoidable_cost * 100
            if total_avoidable_cost > 0
            else 0.0
        )
        comparison_rows.append(
            {
                "category": category,
                "total_cost": data["total_cost"],
                "avg_daily": data["avg_daily_cost"],
                "max_daily": data["max_daily_cost"],
                "min_daily": data["min_daily_cost"],
                "components": len(data["columns"]),
                "share_percent": share,
            }
        )
    comparison_rows = sorted(
        comparison_rows, key=lambda r: r["total_cost"], reverse=True
    )

    trend_fig = go.Figure()
    colors = px.colors.qualitative.Set1[: len(results)]
    for i, (category, data) in enumerate(results.items()):
        daily_data = data["daily_data"].copy()
        daily_data["Date"] = pd.to_datetime(daily_data["Entry Dt"])
        daily_data = daily_data.sort_values("Date")
        trend_fig.add_trace(
            go.Scatter(
                x=daily_data["Date"],
                y=daily_data["Total"],
                mode="lines+markers",
                name=category,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f"{category}<br>"
                    "Date: %{x}<br>"
                    "Cost: ₹%{y:,.0f}<extra></extra>"
                ),
            )
        )
    trend_fig.update_layout(
        title="📈 Daily Avoidable Costs Trend by Category",
        xaxis_title="Date",
        yaxis_title="Cost (₹)",
        height=500,
        hovermode="x unified",
    )
    trend_fig.update_yaxes(tickformat=",.0f", tickprefix="₹")

    return {
        "metrics": {
            "total_avoidable_cost": total_avoidable_cost,
            "total_avg_daily_cost": total_avg_daily,
            "categories_analyzed": len(results),
        },
        "summary_chart": summary_chart,
        "trend_chart": trend_fig.to_dict(),
        "categories": categories_payload,
        "comparison_table": comparison_rows,
        "category_order": list(results.keys()),
    }


# -----------------------------
#   Page-level payload builders (for React tabs)
# -----------------------------

def get_overview_payload() -> Dict[str, Any]:
    data = load_data()
    cost_metrics_global = calculate_cost_metrics(data, vendor_filter=None)
    cost_metrics_ntpc = calculate_cost_metrics(data, vendor_filter="NTPC")

    return {
        "cost_metrics_global": cost_metrics_global,
        "cost_metrics_ntpc": cost_metrics_ntpc,
        "benchmarks_summary": {
            "cil": benchmark_loader.get_cil_benchmark(),
            "sccl": benchmark_loader.get_sccl_benchmark(),
            "domestic_avg": benchmark_loader.get_domestic_coal_average(),
            "international_avg": benchmark_loader.get_international_average(),
        },
    }


def get_benchmarks_payload() -> Dict[str, Any]:
    return {
        "table": benchmark_loader.get_benchmark_table(),
        "cil": benchmark_loader.get_cil_benchmark(),
        "sccl": benchmark_loader.get_sccl_benchmark(),
        "domestic_avg": benchmark_loader.get_domestic_coal_average(),
        "international_avg": benchmark_loader.get_international_average(),
        "delivered_price": benchmark_loader.get_delivered_price_benchmark(),
    }


# Example FastAPI skeleton (optional)

# -----------------------------
#   New April Analysis & KPI Functions
# -----------------------------

def get_april_cost_summary() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    # Filter for purchases only
    if 'Customer Name' in receipt.columns:
        receipt = receipt[receipt['Customer Name'].isna()]
    
    # Calculate Metrics
    total_qty = float(receipt["GR Qty"].sum())
    total_coal_cost = float(receipt["Gross Total (Coal_Z_TOT) P Val"].sum())
    total_freight_cost = float(receipt["Gross Total (Rail_Z_TOT) P Val"].sum())
    total_invoice_value = total_coal_cost + total_freight_cost
    
    avg_landed_cost = (total_invoice_value / total_qty) if total_qty > 0 else 0
    avg_coal_cost = (total_coal_cost / total_qty) if total_qty > 0 else 0
    avg_freight_cost = (total_freight_cost / total_qty) if total_qty > 0 else 0
    
    return {
        "success": True,
        "metrics": {
            "avg_landed_cost": avg_landed_cost,
            "avg_coal_cost": avg_coal_cost,
            "avg_freight_cost": avg_freight_cost,
            "total_quantity": total_qty,
            "total_invoice_value": total_invoice_value
        }
    }

def get_cost_variants() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    total_qty = receipt["GR Qty"].sum()
    
    variants = {
        "Basic Cost per MT": "Basic Val (Coal_ZBC0) P Val",
        "Taxable Coal Cost per MT": "Taxable Coal Cost (Z_TOT) P Val",
        "Gross Coal Cost per MT": "Gross Total (Coal_Z_TOT) P Val",
        "Freight Cost per MT": "Gross Total (Rail_Z_TOT) P Val",
        "Landed Cost per MT": None # Sum of coal and rail
    }
    
    rows = []
    for label, col in variants.items():
        if col and col in receipt.columns:
            val = float(receipt[col].sum() / total_qty) if total_qty > 0 else 0
            rows.append({"variant": label, "avg_cost": val})
        elif label == "Landed Cost per MT":
            coal_col = "Gross Total (Coal_Z_TOT) P Val"
            rail_col = "Gross Total (Rail_Z_TOT) P Val"
            if coal_col in receipt.columns and rail_col in receipt.columns:
                total_val = receipt[coal_col].sum() + receipt[rail_col].sum()
                val = float(total_val / total_qty) if total_qty > 0 else 0
                rows.append({"variant": label, "avg_cost": val})
                
    return {"success": True, "variants": rows}

def get_coal_vs_freight_pie() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    total_coal = float(receipt["Gross Total (Coal_Z_TOT) P Val"].sum())
    total_rail = float(receipt["Gross Total (Rail_Z_TOT) P Val"].sum())
    
    fig = go.Figure(data=[go.Pie(labels=["Coal Cost", "Freight Cost"], 
                                  values=[total_coal, total_rail],
                                  hole=.3,
                                  marker_colors=['#3498db', '#e74c3c'])])
    fig.update_layout(title="Coal Cost vs Freight Cost Breakdown", height=400)
    
    return {"success": True, "chart": fig.to_dict()}

def get_grade_vs_benchmark() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    if 'IV Grade' not in receipt.columns:
        return {"success": False, "error": "IV Grade column missing"}
        
    target_col = 'Gross Total (Coal_Z_TOT) P Val'
    if target_col not in receipt.columns:
        # Fallback to whatever matches roughly
        cols = [c for c in receipt.columns if 'Coal' in c and 'Total' in c]
        if cols: target_col = cols[0]
        else: return {"success": False, "error": f"Target column {target_col} not found"}

    grade_analysis = receipt.groupby('IV Grade').agg({
        'GR Qty': 'sum',
        target_col: 'sum'
    }).reset_index()

    # Remove grades with zero or missing quantity to avoid division errors
    grade_analysis = grade_analysis[grade_analysis['GR Qty'].fillna(0) > 0]
    if grade_analysis.empty:
        return {"success": False, "error": "No valid grade data (zero or missing GR Qty)"}

    grade_analysis['Actual Cost'] = grade_analysis[target_col] / grade_analysis['GR Qty']
    
    # Mock benchmark RP for grades - in reality this would come from a reference table
    grade_benchmarks = {
        'G10': 1500, 'G11': 1400, 'G12': 1300, 'G13': 1200, 'G6': 2200, 'G7': 2000, 'G8': 1800, 'G9': 1600
    }
    
    grade_analysis['Benchmark RP'] = grade_analysis['IV Grade'].map(grade_benchmarks).fillna(1391)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grade_analysis['IV Grade'], y=grade_analysis['Actual Cost'], name='Actual Cost'))
    fig.add_trace(go.Scatter(x=grade_analysis['IV Grade'], y=grade_analysis['Benchmark RP'], 
                             name='Benchmark RP', mode='lines+markers', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title="Grade-wise Coal Cost vs Benchmark", yaxis_title="Cost per MT (₹)", height=400)
    
    # Ensure returned table has exact keys the frontend expects
    table_df = grade_analysis.copy()
    # Keep/rename only the expected columns
    expected_cols = ['IV Grade', 'Actual Cost', 'Benchmark RP']
    for c in expected_cols:
        if c not in table_df.columns:
            table_df[c] = None

    table_records = table_df[expected_cols].to_dict(orient='records')

    return {
        "success": True,
        "table": table_records,
        "chart": fig.to_dict()
    }

def get_cost_breakdown_variants() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    total_qty = receipt["GR Qty"].sum()
    
    breakdown = {
        "Basic": "Basic Val (Coal_ZBC0) P Val",
        "Taxable": "Taxable Coal Cost (Z_TOT) P Val",
        "Gross Coal": "Gross Total (Coal_Z_TOT) P Val",
        "Gross Rail": "Gross Total (Rail_Z_TOT) P Val"
    }
    
    results = {}
    for label, col in breakdown.items():
        if col in receipt.columns:
            results[label] = float(receipt[col].sum() / total_qty) if total_qty > 0 else 0
    
    # Dynamic calculation for GST
    gross_coal = receipt["Gross Total (Coal_Z_TOT) P Val"].sum() if "Gross Total (Coal_Z_TOT) P Val" in receipt.columns else 0
    taxable_coal = receipt["Taxable Coal Cost (Z_TOT) P Val"].sum() if "Taxable Coal Cost (Z_TOT) P Val" in receipt.columns else 0
    results["GST Coal"] = float((gross_coal - taxable_coal) / total_qty) if total_qty > 0 else 0
    
    gross_rail = receipt["Gross Total (Rail_Z_TOT) P Val"].sum() if "Gross Total (Rail_Z_TOT) P Val" in receipt.columns else 0
    results["GST Rail"] = float((gross_rail * 0.05) / total_qty) if total_qty > 0 else 0
    
    results["Landed"] = results.get("Gross Coal", 0) + results.get("Gross Rail", 0)
    
    return {"success": True, "breakdown": results}

def get_vendor_kpis() -> Dict[str, Any]:
    data = load_data()
    if not data: return {"success": False, "error": "Data load failed"}
    
    receipt = data["coal_receipt"]
    gcv = data["gcv_data"]
    
    # Vendor KPIs
    vendor_groups = receipt.groupby('Coal Vendor').agg({'GR Qty': 'sum', 'Gross Total (Coal_Z_TOT) P Val': 'sum'}).reset_index()
    vendor_groups['Cost per MT'] = vendor_groups['Gross Total (Coal_Z_TOT) P Val'] / vendor_groups['GR Qty']
    
    highest_volume_vendor = vendor_groups.loc[vendor_groups['GR Qty'].idxmax()]['Coal Vendor']
    lowest_cost_vendor = vendor_groups.loc[vendor_groups['Cost per MT'].idxmin()]['Coal Vendor']
    
    # Best GCV vendor (requires joining with GCV data if vendor info is there)
    # For now, return mock/simplified KPIs based on available data
    
    return {
        "success": True,
        "kpis": {
            "highest_volume_vendor": highest_volume_vendor,
            "lowest_cost_vendor": lowest_cost_vendor,
            "best_gcv_vendor": "NTPC Siding", # Mock till GCV join is fixed
            "most_consistent_vendor": "SECL",
            "best_day": "2024-04-12"
        }
    }


def get_cost_sunburst(receipt: pd.DataFrame = None) -> Dict[str, Any]:
    """Build a sunburst chart (Coal Vendor -> IV Grade) using GR Qty as values.

    If `receipt` is provided, use it; otherwise load from files.
    """
    if receipt is None:
        data = load_data()
        if not data:
            return {"success": False, "error": "Data load failed"}
        receipt = data["coal_receipt"]

    # normalize columns
    receipt = receipt.copy()
    receipt.columns = receipt.columns.str.strip()

    if not all(c in receipt.columns for c in ["Coal Vendor", "IV Grade", "GR Qty"]):
        return {"success": False, "error": "Required columns for sunburst missing"}

    fig = px.sunburst(
        receipt,
        path=["Coal Vendor", "IV Grade"],
        values="GR Qty",
        title="Coal Volume Distribution by Vendor and Grade",
    )
    return {"success": True, "chart": fig.to_dict()}


def get_vendor_sunburst(receipt: pd.DataFrame = None) -> Dict[str, Any]:
    """Alternative vendor-focused sunburst (Vendor -> Month -> Grade).

    If `receipt` provided, use it; otherwise load from files.
    """
    if receipt is None:
        data = load_data()
        if not data:
            return {"success": False, "error": "Data load failed"}
        receipt = data["coal_receipt"].copy()

    receipt = receipt.copy()
    receipt.columns = receipt.columns.str.strip()
    if "Entry Dt" in receipt.columns:
        receipt["Month"] = pd.to_datetime(receipt["Entry Dt"]).dt.to_period("M").astype(str)
    else:
        receipt["Month"] = "Unknown"

    if not all(c in receipt.columns for c in ["Coal Vendor", "IV Grade", "GR Qty"]):
        return {"success": False, "error": "Required columns for vendor sunburst missing"}

    fig = px.sunburst(
        receipt,
        path=["Coal Vendor", "Month", "IV Grade"],
        values="GR Qty",
        title="Vendor -> Month -> Grade Distribution",
    )
    return {"success": True, "chart": fig.to_dict()}
