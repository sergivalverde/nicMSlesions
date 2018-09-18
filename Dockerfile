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
RUN rm /Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# install CNN related packages
ADD requirements.txt /requirements.txt
RUN conda install numpy scipy mkl
RUN pip install pip --upgrade
RUN pip install tensorflow-gpu==1.6.0
RUN pip install -r /requirements.txt

# create a docker user
RUN useradd -ms /bin/bash docker
ENV HOME /home/docker

# copy necessary files to container
RUN mkdir $HOME/src
RUN mkdir /data
ENV PATH=/$HOME/src:${PATH}
ADD __init__.py $HOME/src/
ADD .keras $HOME/.keras
ADD app.py $HOME/src/
ADD cnn_scripts.py $HOME/src/
# ADD config $HOME/src/config
# ADD nets $HOME/src/nets
ADD libs $HOME/src/libs
ADD utils $HOME/src/utils
ADD logonic.png $HOME/src/
ADD nic_train_network_batch.py $HOME/src/
ADD nic_infer_segmentation_batch.py $HOME/src/

# add permissions (odd)
# RUN chown docker -R nets
# RUN chown docker -R config

USER docker
WORKDIR $HOME/src
