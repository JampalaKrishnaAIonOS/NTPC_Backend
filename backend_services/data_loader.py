from pathlib import Path
import os
import pandas as pd

# Determine DATA_ROOT in a portable way:
# Priority: 1) environment variable `NTPC_DATA_ROOT`
#           2) existing `C:\ntpc_data` (legacy/hardcoded)
#           3) the repository `Data` folder (sibling of project root)
env_root = os.environ.get("NTPC_DATA_ROOT")
if env_root:
    DATA_ROOT = Path(env_root)
else:
    legacy = Path(r"C:\ntpc_data")
    if legacy.exists():
        DATA_ROOT = legacy
    else:
        # backend_services/.. -> project root, Data folder lives at project_root / 'Data'
        DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

print(f"DATA_ROOT set to: {DATA_ROOT}")

def load_data():
    return {
        "coal_stock": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Daywise_CoalStock"),
        "coal_consumption": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Daywise_CoalConsumption"),
        "coal_receipt": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Daywise_CoalReceipt_C&FCost"),
        "gcv_data": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Daywise_GCVData"),
        "siding_details": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Siding Details"),
        "mine_code": pd.read_excel(DATA_ROOT / "Dadri_Data_April.xlsx", sheet_name="Mine Code"),
    }
