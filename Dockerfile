FROM python:3.8.6

ADD . /Fase_detection

WORKDIR /Fase_detection

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
