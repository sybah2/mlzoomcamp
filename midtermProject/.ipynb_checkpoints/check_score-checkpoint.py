import requests

url = 'http://localhost:9696/predict'


client = {'credit_score': 626,
 'country': 'France',
 'gender': 'Female',
 'age': 29,
 'tenure': 4,
 'balance': 105767.28,
 'products_number': 2,
 'credit_card': 0,
 'active_member': 0,
 'estimated_salary': 41104.82}

score = requests.post(url, json=client).json()

print(score)

churn = score['Churn']
probability_of_churn = score['Proberbility of churn']

if churn == True:
    print(f'This client will likely churn with churn probability of {probability_of_churn}. Send a promo email')
else:
    print(f'The client is less likely to churn with a porbality of {probability_of_churn}. No need to send an email')
