## Machine learning zoomcamp 2024 Midterm project

### Summary
This project is the midterm project for machine learning zoomcamp 2024. Banks provide services such credits cards to customers and would like for customers to stay for the long term. However often time customers leave the bank (`Churn`) and join other banks. It is important that banks analyse customers who could potentially churn and give the better offers to stay with the Bank.
In this project a machine learning approach is use to determine if a bank customer will churn so that customer servive can send the promotions. Different models are trained and different parameter tunes for each model to determine the best parameters. The data is from 3 European countries; France Spain and Germany. 

### Description of data
The used for the analysis was collected from 3 European countries, France, Spain and Germany and contain the following variables:

- customer_id:	`Customer unique identifier`
- credit_score:	`Customer credit score`
- country:	`Customer’s country of residence`
- gender:	`Customer’s gender`
- age:	`Customer’s age`
- tenure:	`Number of years the customer has been with the bank`
- balance	`Customer account Balance`
- products_number:	`Number of products used by the customer`
- credit_card:	`Does customer has a credit card (1: Yes, 0: No)`
- active_member:	`Is the customer an active member (1: Yes, 0: No)`
- estimated_salary:	`Customer estimated annual salary`
- churn:	`Does this customer churned (1: Yes, 0: No)`

### Folder Struture and content
- The `data` folder contain the data used for the analysis and contains `bank_churn_data.csv`
- `notebook.ipynb` contains the exploratory data analysis (EDA), training diffrent models and their tunning
- `final_model_training.ipynb` contains the final model training
- `train.py` is the python script thata does the training of the model
- `predict.py` is use for the predicting the potential of a customer churning


### How to run the classification

