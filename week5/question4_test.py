import requests

url = 'http://localhost:9696/churn'


client = {"job": "student", "duration": 280, "poutcome": "failure"}
score = requests.post(url, json=client).json()

print(score)