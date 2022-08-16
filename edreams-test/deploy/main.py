import requests
import json
import numpy as np

url ="http://localhost:8501/v1/models/extra_baggage:predict"
data = {
    "inputs": {
        "examples": [b'\n\xbb\x02\n\x18\n\tDEPARTURE\x12\x0b\n\t\n\x0722/July\n\x16\n\x06DEVICE\x12\x0c\n\n\n\x08COMPUTER\n\x0c\n\x03GDS\x12\x05\x1a\x03\n\x01\x01\n\x1a\n\tTRIP_TYPE\x12\r\n\x0b\n\tTRIP_TYPE\n\x0f\n\x03SMS\x12\x08\n\x06\n\x04TRUE\n\x13\n\x07PRODUCT\x12\x08\n\x06\n\x04TRIP\n\x14\n\x08DISTANCE\x12\x08\x12\x06\n\x04\xb8nHE\n\x0f\n\x06NO_GDS\x12\x05\x1a\x03\n\x01\x00\n\x19\n\tHAUL_TYPE\x12\x0c\n\n\n\x08DOMESTIC\n\x16\n\x07ARRIVAL\x12\x0b\n\t\n\x0722/July\n\x12\n\x05TRAIN\x12\t\n\x07\n\x05FALSE\n\x11\n\x08CHILDREN\x12\x05\x1a\x03\n\x01\x00\n\x13\n\x07WEBSITE\x12\x08\n\x06\n\x04EDES\n\x10\n\x07INFANTS\x12\x05\x1a\x03\n\x01\x00\n\x0f\n\x06ADULTS\x12\x05\x1a\x03\n\x01\x01']
    }
}
response = requests.post(url, data=data, headers={"content-type" : "application/json"})
print(response.text)
predictions = json.loads(response.text)

print(predictions)