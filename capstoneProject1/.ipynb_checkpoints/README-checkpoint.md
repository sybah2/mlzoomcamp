## Machine learning zoomcamp 2024 Capstone project

![T](images/diamond.jpeg)


### Summary
This capstone project of the machine learning zoomcamp 2024. Diamond are precious and determining their price is crucial.
In the price different machine learning models were used to predict the price of a given diamond based on the features of the data describe below. 


### Description of data
The decription of the diamond can be found below
- carat: `weight of the diamond `
- cut: `quality of cut (Fair, Good, Very Good, Premium, Ideal)`
- color: `diamond colour, from J (worst) to D (best)`
- clarity: `a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))`
- depth: `total depth percentage = z / mean(x, y) = 2 * z / (x + y)`
- table: `width of top of diamond relative to widest point`
- price: `price of diamond in dolars`
- x: `lenght in mm`
- y: `width in mm`
- z: `depth in mm`


### Folder Struture and their content
The imporant folders and file are detailled below
- The `data` folder contain the data used for the analysis and contains `bank_churn_data.csv`
- `notebook.ipynb` contains the exploratory data analysis (EDA), training diffrent models and their tunning
- `final_model.ipynb` contains the final model training
- `train.py` is the python script thata does the training of the model
- `predict.py` is use for the predicting the potential of a customer churning
- `Dockerfile` is docker file for building the docker container needed for the churning services
- `Pipfile.lock` and `Pipfile` contain the pipenv specification for the packages needed to run the churning services
- `predict.py` script containing the functions for predictions
- `final_model.bin` the final model needed for prediction
- `check_price.py` The script for checking the price of a diamond based on its details. `NB: this need to be updated with the new client details.` 


### How to predict the price of a diamond
- Download and install Docker
To run this project, user need to istall docker for portability and reproducibility. The docker image contain all the packages needed for the project. The isntallation instructions can me found [here](https://docs.docker.com/get-started/get-docker/)

- Build the docker image
Following the downlaod and installation of docker, run the command below to build the docker image.

`docker build --no-cache -t diamond_price_service .`

- Lunch the churning service
Once succesfully docker image build do the churning service can be started using the command below
`docker run -it --rm -p 9696:9696 diamond_price_service`


- Update diamond specifications

One the diamond_price_service is running, open the check_score.py and change diamond details below to reflect the diamond and save the python script.

Details of the diamond can be updated to predict its price

`diamond = {
'carat': 1.12,
 'cut': 3,
 'color': 4,
 'clarity': 1,
 'depth': 60.5,
 'table': 59.0,
 'x': 6.79,
 'y': 6.73,
 'z': 4.09}
 `

- Determine the price of a diamond
`python check_price.py`

After runing this script the printed output will give probability of a client churning and weather a promotional email should be sent or not. 