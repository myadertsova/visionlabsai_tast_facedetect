FROM python:3.8.6

ADD . /Fase_detection

WORKDIR /Fase_detection

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip3 install -r requirements.txt

EXPOSE 8080
