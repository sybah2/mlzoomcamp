import requests

url = 'http://localhost:9696/predict'


client = {"job": "management", "duration": 400, "poutcome": "success"}
score = requests.post(url, json=client).json()

print(score)