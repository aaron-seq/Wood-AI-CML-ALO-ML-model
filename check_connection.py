import requests

def check_api_connection():
    try:
        response = requests.get('http://localhost:8000/health', timeout=2)
        return response.status_code == 200
    except:
        return False

if check_api_connection():
    print('API is accessible!')
else:
    print('API is NOT accessible from dashboard')
