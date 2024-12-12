import json

import requests

url = "http://localhost:7000/process"

data = {"query":"Need to know about Kevin Owocki",
        "uid":"Karthik7"}

headers = {"Content-type": "application/json"}

with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunk in r.iter_content(4096):
        print(chunk.decode("utf-8"))