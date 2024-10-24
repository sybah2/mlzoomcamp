import requests

url = 'http://127.0.0.1:9696/predict'

customer = 'abz_123'

client = {"job": "student", "duration": 280, "poutcome": "failure"}
score = requests.post(url, json=client).json()

print(score)