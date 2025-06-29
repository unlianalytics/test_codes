import requests

url = 'http://127.0.0.1:8000/api/lte-parameters'
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print('--- LTE Parameters API Test ---')
    print(f"Managed Objects: {len(data.get('unique_managed_objects', []))}")
    print(f"First 5 Managed Objects: {data.get('unique_managed_objects', [])[:5]}")
    print(f"IDs: {len(data.get('unique_ids', []))}")
    print(f"First 5 IDs: {data.get('unique_ids', [])[:5]}")
    print(f"Results: {len(data.get('results', []))}")
    if data.get('results'):
        print('Sample result:')
        for k, v in data['results'][0].items():
            print(f"  {k}: {v}")
    else:
        print('No results found.')
except Exception as e:
    print('Error testing LTE API:', e) 