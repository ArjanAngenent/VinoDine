FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements_prod.txt requirements.txt

RUN pip install -r requirements.txt

COPY vinodine vinodine

COPY setup.py setup.py

COPY model.pkl model.pkl

RUN pip install .

CMD uvicorn vinodine.api.fast:app --host 0.0.0.0 --port $PORT
