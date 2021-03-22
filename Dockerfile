FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home
RUN apt-get update && apt-get upgrade -y

COPY . .

RUN apt-get install -y \
    curl \
    g++ \
    cmake \
    libboost-all-dev \
    python3.7 \
    python3-pip

RUN pip3 install \
    tqdm==4.48.2 \
    numpy==1.18.1 \
    pandas==1.0.3 \
    matplotlib==3.1.3 \
    seaborn==0.10.1

RUN pip3 install GPy==1.9.9

RUN cmake -B build && cmake --build build
