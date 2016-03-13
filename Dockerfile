FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python2.7 \
    python-setuptools \
    python-pip \
    python-numpy \
    python-scipy \
    python-matplotlib

ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt

ADD src /src

EXPOSE  5000

CMD [ "python", "./src/start_service.py" ]
