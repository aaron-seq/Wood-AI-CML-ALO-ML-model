import requests

print('Testing API...')

# Health check
health = requests.get('http://localhost:8000/health').json()
print(f"Health: {health['status']}")

# Score data
with open('data/cml_sample_500.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/score-cml-data', files={'file': f})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Scored {result['rows_scored']} CMLs")
    else:
        print(f"Error: {response.text}")
