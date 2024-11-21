FROM python:3.8.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libpq-dev \
    && apt-get clean
 
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
 
RUN pipenv install --system --deploy --verbose

 
COPY ["predict.py", "xboos_model.bin", "./"]
 
EXPOSE 9696
 
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]