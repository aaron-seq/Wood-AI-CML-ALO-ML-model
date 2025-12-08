import requests
import pandas as pd

print("Generating predictions via API...")

# Score all CMLs
with open("data/cml_sample_500.csv", "rb") as f:
    response = requests.post("http://localhost:8000/score-cml-data", files={"file": f})
    result = response.json()

# Create detailed results
df = pd.DataFrame(result["results"])

print("
" + "="*70)
print("PREDICTION RESULTS")
print("="*70 + "
")

print(f"Total CMLs Analyzed: {result['rows_scored']}")
print(f"Model: {result['model_info']['model_type']}
")

# Summary
eliminate = (df["recommendation"] == "ELIMINATE").sum()
keep = (df["recommendation"] == "KEEP").sum()
print("RECOMMENDATIONS:")
print(f"  ELIMINATE: {eliminate} CMLs ({eliminate/len(df)*100:.1f}%)")
print(f"  KEEP: {keep} CMLs ({keep/len(df)*100:.1f}%)
")

# Confidence
high = (df["confidence"] == "HIGH").sum()
mod = (df["confidence"] == "MODERATE").sum()
print("CONFIDENCE LEVELS:")
print(f"  HIGH: {high} ({high/len(df)*100:.1f}%)")
print(f"  MODERATE: {mod} ({mod/len(df)*100:.1f}%)
")

# Probability stats
print("PROBABILITY STATISTICS:")
print(f"  Mean: {df['elimination_probability'].mean():.2%}")
print(f"  Median: {df['elimination_probability'].median():.2%}")
print(f"  Std Dev: {df['elimination_probability'].std():.2%}
")

# Top elimination candidates
print("TOP 10 ELIMINATION CANDIDATES:")
for i, row in df.nlargest(10, "elimination_probability").iterrows():
    print(f"  {row['id_number']}: {row['elimination_probability']:.1%} - {row['confidence']} confidence")

print("
TOP 10 CMLS TO KEEP:")
for i, row in df.nsmallest(10, "elimination_probability").iterrows():
    print(f"  {row['id_number']}: {row['elimination_probability']:.1%} - {row['confidence']} confidence")

# Cost analysis
annual_savings = eliminate * 500 * 2
print("
" + "="*70)
print("COST SAVINGS ANALYSIS")
print("="*70)
print(f"Eliminated CMLs: {eliminate}")
print(f"Cost per inspection: 500")
print(f"Inspections per year: 2")
print(f"Annual savings: {annual_savings:,}")
print(f"5-year savings: {annual_savings*5:,}")
print(f"10-year savings: {annual_savings*10:,}
")

# Save results
df.to_csv("dashboard_predictions.csv", index=False)
print("Results saved to: dashboard_predictions.csv")
print("="*70)
