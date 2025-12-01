import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your realistic MSME data
df = pd.read_csv("msme_credit_data.csv")

# Select features for training (as per RBI/fintech practice)
features = [
    "business_age_years",
    "monthly_revenue_lakhs",
    "transaction_frequency_per_month",
    "avg_settlement_days",
    "gst_compliance_score",
    "payment_consistency_score"
]
X = df[features]
y = df["cashflow_approved"]  # Target: cash-flow model approval

# Train Random Forest
model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X, y)

# Save the model (creates model.pkl for FastAPI)
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
