import requests
import json

print("\n--- TEST 6: Verify Port 8000 ---")
try:
    with open("data/sample_cml_data.csv", "rb") as f:
        file_content = f.read()

    files = {"file": ("sample_cml_data.csv", file_content, "text/csv")}
    # Using 8000
    response = requests.post("http://localhost:8000/upload-cml-data", files=files)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print("Message:", data.get("message"))
        print("Content:", json.dumps(data, indent=2))
    else:
        print("Error response:", response.text)

except Exception as e:
    print(f"Script Error: {e}")
