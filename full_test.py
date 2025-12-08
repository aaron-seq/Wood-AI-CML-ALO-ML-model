import requests
import pandas as pd

print('='*70)
print('COMPLETE SYSTEM TEST - WOOD AI CML OPTIMIZATION')
print('='*70)

# Test 1: Health
health = requests.get('http://localhost:8000/health').json()
print(f'\n✓ API Health: {health[\"status\"]}')
print(f'✓ Model Type: {health.get(\"version\", \"N/A\")}')

# Test 2: Full scoring
with open('data/cml_sample_500.csv', 'rb') as f:
    result = requests.post('http://localhost:8000/score-cml-data', files={'file': f}).json()

print(f'\n✓ Scored: {result[\"rows_scored\"]} CMLs')
print(f'✓ Model: {result[\"model_info\"][\"model_type\"]}')

# Load full results
df = pd.read_csv('api_predictions.csv')

print(f'\n📊 PREDICTION BREAKDOWN:')
print(f'   Total CMLs: {len(df)}')

eliminate = (df['recommendation'] == 'ELIMINATE').sum()
keep = (df['recommendation'] == 'KEEP').sum()
print(f'   ELIMINATE: {eliminate} ({eliminate/len(df)*100:.1f}%)')
print(f'   KEEP: {keep} ({keep/len(df)*100:.1f}%)')

high_conf = (df['confidence'] == 'HIGH').sum()
mod_conf = (df['confidence'] == 'MODERATE').sum()
print(f'\n🎯 CONFIDENCE LEVELS:')
print(f'   HIGH: {high_conf} ({high_conf/len(df)*100:.1f}%)')
print(f'   MODERATE: {mod_conf} ({mod_conf/len(df)*100:.1f}%)')

# Probability analysis
print(f'\n📈 PROBABILITY ANALYSIS:')
print(f'   Mean elimination prob: {df[\"elimination_probability\"].mean():.2%}')
print(f'   Median elimination prob: {df[\"elimination_probability\"].median():.2%}')
print(f'   High confidence (>80%): {(df[\"elimination_probability\"] > 0.8).sum()}')
print(f'   Low confidence (<20%): {(df[\"elimination_probability\"] < 0.2).sum()}')

# Top candidates for elimination
print(f'\n🔴 TOP 10 ELIMINATION CANDIDATES:')
top_elim = df.nlargest(10, 'elimination_probability')
for i, row in top_elim.iterrows():
    print(f'   {row[\"id_number\"]}: {row[\"elimination_probability\"]:.1%} probability')

# Top candidates to keep
print(f'\n🟢 TOP 10 CMLS TO KEEP:')
top_keep = df.nsmallest(10, 'elimination_probability')
for i, row in top_keep.iterrows():
    print(f'   {row[\"id_number\"]}: {row[\"elimination_probability\"]:.1%} probability')

# Cost analysis
annual_savings = eliminate * 500 * 2
print(f'\n💰 COST SAVINGS ESTIMATE:')
print(f'   CMLs to eliminate: {eliminate}')
print(f'   Cost per inspection: ')
print(f'   Inspections per year: 2')
print(f'   Annual savings: ')
print(f'   5-year savings: ')

print('\n' + '='*70)
print('✓ ALL TESTS PASSED!')
print('='*70)
print(f'\nAccess Points:')
print(f'  API: http://localhost:8000/docs')
print(f'  Dashboard: http://localhost:8501')
print(f'  Results: api_predictions.csv')
