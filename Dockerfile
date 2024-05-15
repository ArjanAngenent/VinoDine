FROM tensorflow/tensorflow:2.10.0

WORKDIR /prod

COPY requirements.txt requirements.txt

COPY vinodine vinodine

COPY model.py model.py

RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py setup.py

#the actual name of the Python script
CMD uvicorn vinodine.api.fast:app --host 0.0.0.0 --port $PORT
