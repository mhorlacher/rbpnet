FROM python:3.8

COPY . /rbpnet

# install rbpnet
RUN pip install /rbpnet/

# input/output directory for biolib
RUN mkdir /inputs
RUN mkdir /outputs
