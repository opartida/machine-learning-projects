from unicodedata import name
import requests
import json
import numpy as np
import tensorflow as tf
import base64

url ="http://localhost:8501/v1/models/extra_baggage:predict"
import tensorflow as tf
import json
features = {
      "departure": ["22/July"],
      "arrival": ["22/July"],
    }

data = {    
    'signature_name': 'serving_rest', 
    'instances': [features]
    }
data = json.dumps(data)

print(data)

response = requests.post(url, data=data, headers={"content-type" : "application/json"})
print(response.text)
predictions = json.loads(response.text)
