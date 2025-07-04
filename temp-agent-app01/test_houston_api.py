import requests
import json

def test_houston_site_names():
    try:
        response = requests.get('http://localhost:8000/api/houston-site-names')
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data type: {type(data)}")
            print(f"Data length: {len(data) if isinstance(data, list) else 'Not a list'}")
            print(f"First 5 items: {data[:5] if isinstance(data, list) else data}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_houston_site_names() 