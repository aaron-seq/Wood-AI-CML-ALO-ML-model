import requests
import json

print("Testing API response...")

with open('data/cml_sample_500.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/score-cml-data', files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse Content:")
    
    try:
        result = response.json()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response.text[:500]}")
