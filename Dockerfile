FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Sergi Valverde <svalverde@eia.udg.edu>

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  #python-pip \
  #python-nose \
  #python-numpy \
  #python-scipy \
  python-tk\
  git \
  curl

# git CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# install theano and pygpu
RUN conda install numpy scipy mkl
RUN conda install theano pygpu
USER root

# CNN necessary files
ADD .theanorc /root/.theanorc
ADD app.py /app.py
ADD cnn_scripts.py /cnn_scripts.py
ADD config /config
ADD libs /libs
ADD logonic.png /logonic.png
ADD nets /nets
ADD requirements.txt /requirements.txt
ADD utils /utils

# install packages
RUN pip install pip --upgrade
RUN pip install -r requirements.txt

CMD python app.py
