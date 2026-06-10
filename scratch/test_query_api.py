import json
import urllib.request

url = "http://localhost:8000/api/v1/query"
payload = {
    "query": "Lương đi làm ngày Tết tính thế nào, nếu lương thường là 500k?",
    "strategy": "hybrid_rerank",
    "top_k": 5,
    "bypass_refusal": False
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Sending POST request to RAG query endpoint with math question...")
try:
    with urllib.request.urlopen(req) as response:
        res_data = response.read().decode("utf-8")
        parsed = json.loads(res_data)
        
        # Write to JSON file
        with open("scratch/query_response_math.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print("Success! Response saved to scratch/query_response_math.json")
except Exception as e:
    print(f"Error calling query endpoint: {e}")
