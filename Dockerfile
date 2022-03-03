From python:3.8.12-buster


COPY requirements.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY ui_customerchurn /ui_customerchurn
COPY app.py /app.py

CMD streamlit run app.py --server.port $PORT
