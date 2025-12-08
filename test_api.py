import requests
import pandas as pd

print("=" * 60)
print("WOOD AI CML OPTIMIZATION - API TEST")
print("=" * 60)

# Test 1: Health Check
print("\n1. Health Check:")
response = requests.get('http://localhost:8000/health')
health = response.json()
print(f"   Status: {health['status']}")
print(f"   Model Loaded: {health['model_loaded']}")
print(f"   Version: {health['version']}")

# Test 2: Score CML Data
print("\n2. Scoring CML Data:")
with open('data/cml_sample_500.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/score-cml-data', files=files)
    result = response.json()

print(f"   Total CMLs Scored: {result['rows_scored']}")
print(f"   Model Type: {result['model_info']['model_type']}")

# Analyze predictions
results_df = pd.DataFrame(result['results'])
eliminate_count = (results_df['predicted_elimination_flag'] == 1).sum()
keep_count = (results_df['predicted_elimination_flag'] == 0).sum()

print(f"\n3. Prediction Summary:")
print(f"   ELIMINATE: {eliminate_count} CMLs ({eliminate_count/len(results_df)*100:.1f}%)")
print(f"   KEEP: {keep_count} CMLs ({keep_count/len(results_df)*100:.1f}%)")

# Confidence breakdown
high_conf = (results_df['confidence'] == 'HIGH').sum()
moderate_conf = (results_df['confidence'] == 'MODERATE').sum()

print(f"\n4. Confidence Distribution:")
print(f"   HIGH: {high_conf} ({high_conf/len(results_df)*100:.1f}%)")
print(f"   MODERATE: {moderate_conf} ({moderate_conf/len(results_df)*100:.1f}%)")

# Sample predictions
print(f"\n5. Sample Predictions (First 10):")
for i, pred in enumerate(result['results'][:10], 1):
    rec = pred['recommendation']
    prob = pred['elimination_probability']
    conf = pred['confidence']
    print(f"   {i:2}. {pred['id_number']}: {rec:10} (prob: {prob:.2%}, conf: {conf})")

# Save full results
results_df.to_csv('api_predictions.csv', index=False)
print(f"\n✓ Full results saved to: api_predictions.csv")

# Cost savings calculation
annual_savings = eliminate_count * 500 * 2  #  per inspection, 2x/year
print(f"\n6. Potential Annual Savings:")
print(f"   Eliminated CMLs: {eliminate_count}")
print(f"   Cost per inspection: ")
print(f"   Inspections per year: 2")
print(f"   Estimated savings: /year")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
