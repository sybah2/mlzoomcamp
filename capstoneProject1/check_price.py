import requests

url = 'http://localhost:9696/predict'


diamond = {'carat': 1.12,
 'cut': 3,
 'color': 4,
 'clarity': 1,
 'depth': 60.5,
 'table': 59.0,
 'x': 6.79,
 'y': 6.73,
 'z': 4.09}

price = requests.post(url, json=diamond).json()

print("\n")
print(f"Properties of the diamond\n{diamond}\n")

print(f"The price of a diamond with these features is:\n${price}")
