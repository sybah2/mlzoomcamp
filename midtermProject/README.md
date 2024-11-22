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
The imporant folders and file are detailled below
- The `data` folder contain the data used for the analysis and contains `bank_churn_data.csv`
- `notebook.ipynb` contains the exploratory data analysis (EDA), training diffrent models and their tunning
- `final_model.ipynb` contains the final model training
- `train.py` is the python script thata does the training of the model
- `predict.py` is use for the predicting the potential of a customer churning
- `Dockerfile` is docker file for building the docker container needed for the churning services
- `Pipfile.lock` and `Pipfile` contain the pipenv specification for the packages needed to run the churning services
- `predict.py` script containing the functions for predictions
- `xboos_model.bin` the final model needed for prediction
- `check_score.py` The script for checking the probability of a bank client churning. `NB: this need to be updated with the new client details.` 

### How to run the classification
- Download and install Docker
To run the project user need to have docker install in their local machine. The isntallation instructions can me found [here](https://docs.docker.com/get-started/get-docker/)

- Build the docker image
After downloading installing docker, run the command below to build the docker image.

`docker build --no-cache -t chrun_service .`

- Lunch the churning service
Once succesfully docker image build do the churning service can be started using the command below
`docker run -it --rm -p 9696:9696 chrun_service`


- Update customer details

One the churn service is running, open the check_score.py and change client (customer) details below to the clients and save the python script.

Details to be changed/updated to reflect customer details


`client = {'credit_score': 626,
 'country': 'France',
 'gender': 'Female',
 'age': 29,
 'tenure': 4,
 'balance': 105767.28,
 'products_number': 2,
 'credit_card': 0,
 'active_member': 0,
 'estimated_salary': 41104.82}
 `

- Check the customer churning probability
python check_score.py