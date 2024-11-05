FROM ubuntu:20.04
LABEL maintainer="rhermary"

WORKDIR /app/
COPY requirements.txt requirements.txt
ENV PYTHONPATH src

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y\
    && apt-get install curl wget software-properties-common build-essential\
                       git -y\
    && add-apt-repository ppa:deadsnakes/ppa -y\
    && apt install python3.10 python3.10-distutils -y\
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1\
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python\
    && python -m pip install --upgrade pip
    
RUN python -m pip install -r requirements.txt
RUN git config --global --add safe.directory "*"
RUN chmod 777 /app/

CMD /bin/bash