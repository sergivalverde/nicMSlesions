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
  python-tk\
  git \
  curl \
  emacs24

USER root
ENV CUDA_ROOT /usr/local/cuda/bin

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# install CNN related packages
ADD requirements.txt /requirements.txt
RUN conda install numpy scipy mkl
RUN conda install theano pygpu
RUN pip install pip --upgrade
RUN pip install -r /requirements.txt

#copy necessary files to container
RUN mkdir /src
ENV PATH=/src:${PATH}
ADD .theanorc /.theanorc
ADD app.py /src/
ADD cnn_scripts.py /src/
ADD config /src/config
ADD nets /src/nets
ADD libs /src/libs
ADD utils /src/utils
ADD logonic.png /src/
ADD nic_train_network_batch.py /src/
ADD nic_infer_segmentation_batch.py /src/
WORKDIR /src
