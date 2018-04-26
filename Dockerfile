FROM python:3.5

RUN apt-get update && apt-get install -y libblas-dev liblapack-dev gfortran

RUN pip install --upgrade pip
RUN pip install numpy scipy

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY image_match /usr/src/app/image_match
COPY setup.py /usr/src/app/setup.py

RUN pip install --no-cache-dir -e .[dev]
