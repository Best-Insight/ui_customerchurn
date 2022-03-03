From python:3.8.12-buster


COPY requirements.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY ui_customerchurn /ui_customerchurn
COPY api /api
COPY models/Financial_Services_model /models/Financial_Services_model
COPY models/finance_encoder /models/finance_encoder

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
