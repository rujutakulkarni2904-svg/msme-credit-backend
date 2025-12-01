from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io

# ----- FastAPI app -----
app = FastAPI(
    title="MSME Cash Flow Credit Scoring API",
    description="Compares traditional collateral approvals vs cash-flow model for Indian MSMEs.",
    version="1.0.0",
)

# Allow frontend (Vercel) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your Vercel domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print("⚠️ Could not load model.pkl:", e)

# Features used for scoring
FEATURE_COLUMNS = [
    "business_age_years",
    "monthly_revenue_lakhs",
    "transaction_frequency_per_month",
    "avg_settlement_days",
    "gst_compliance_score",
    "payment_consistency_score",
]

@app.get("/")
async def root():
    return {"status": "ok", "message": "MSME Cash Flow Credit API running"}

@app.get("/sample-portfolio")
async def sample_portfolio():
    """
    Returns the 10 predefined MSMEs + portfolio metrics (trad vs cash-flow).
    """
    try:
        df = pd.read_csv("msme_credit_data.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read msme_credit_data.csv: {e}")

    trad_approved = int(df["traditional_approved"].sum())
    cash_approved = int(df["cashflow_approved"].sum())
    trad_credit = float(df.loc[df["traditional_approved"] == 1, "sim_loan_amount_lakhs"].sum())
    cash_credit = float(df.loc[df["cashflow_approved"] == 1, "sim_loan_amount_lakhs"].sum())

    return {
        "data": df.to_dict(orient="records"),
        "portfolio_metrics": {
            "traditional_approved_count": trad_approved,
            "cashflow_approved_count": cash_approved,
            "traditional_approval_rate_pct": trad_approved * 10,
            "cashflow_approval_rate_pct": cash_approved * 10,
            "traditional_credit_lakh": round(trad_credit, 1),
            "cashflow_credit_lakh": round(cash_credit, 1),
            "extra_credit_unlocked_lakh": round(cash_credit - trad_credit, 1),
        },
    }

def _auto_fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map messy user columns to our expected feature columns.
    Minimal smart behavior so hiring managers don't see crashes.
    """
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    mapping_rules = {
        "business_age_years": ["age", "business_age", "vintage_years"],
        "monthly_revenue_lakhs": ["revenue", "monthly_revenue", "turnover_lakhs"],
        "transaction_frequency_per_month": ["tx_per_month", "transactions", "txn_count"],
        "avg_settlement_days": ["settlement_days", "dsos", "avg_dso"],
        "gst_compliance_score": ["gst_score", "gst_compliance", "gst"],
        "payment_consistency_score": ["payment_score", "on_time_payments", "consistency_score"],
    }

    for target, aliases in mapping_rules.items():
        if target in df.columns:
            col_map[target] = target
            continue
        found = None
        for alias in aliases:
            for lc, original in lower_cols.items():
                if alias in lc:
                    found = original
                    break
            if found:
                break
        if found:
            col_map[found] = target

    df = df.rename(columns=col_map)
    return df

@app.post("/score")
async def score_portfolio(file: UploadFile = File(...)):
    """
    Accepts a CSV file, auto-fixes column names, runs model, returns risk + approvals.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    df = _auto_fix_columns(df)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns even after auto-fix.",
                "expected_columns": FEATURE_COLUMNS,
                "missing_columns": missing,
            },
        )

    X = df[FEATURE_COLUMNS]

    try:
        probs = model.predict_proba(X)[:, 1]  # probability of approval
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    df["approval_probability"] = probs
    df["risk_score"] = (1 - df["approval_probability"]) * 100  # lower is better
    df["approved"] = (df["risk_score"] < 40).astype(int)

    portfolio = {
        "count": int(len(df)),
        "approved_count
