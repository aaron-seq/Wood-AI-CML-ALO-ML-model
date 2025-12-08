
import requests
from pathlib import Path

def test_score_endpoint():
    print("Testing score endpoint...")
    url = "http://localhost:8000/score-cml-data"
    file_path = Path("data/sample_cml_data.csv")
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            data = response.json()
            
            print("\nResponse Keys:", data.keys())
            if "results" in data:
                print(f"SUCCESS: 'results' key found in response with {len(data['results'])} items.")
                print("Sample result item keys:", data['results'][0].keys())
            else:
                print("FAILURE: 'results' key NOT found in response.")
                
            if "predictions" in data:
                print("WARNING: 'predictions' key found in response (unexpected).")
            else:
                print("CONFIRMED: 'predictions' key NOT found in response (expected).")
                
        except Exception as e:
            print(f"Error calling API: {e}")

if __name__ == "__main__":
    test_score_endpoint()
