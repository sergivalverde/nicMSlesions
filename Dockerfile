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

# add /user
RUN useradd -ms /bin/bash user
ENV HOME /home/user
WORKDIR /home/user

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p $HOME/miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=$HOME/miniconda/bin:${PATH}
RUN conda update -y conda

# install CNN related packages
ADD requirements.txt /requirements.txt
RUN conda install numpy scipy mkl
RUN conda install theano pygpu
RUN pip install pip --upgrade
RUN pip install -r /requirements.txt

# copy necessary files to container
# USER user
RUN mkdir $HOME/src
ENV PATH=$HOME/src:${PATH}
ADD .theanorc $HOME/.theanorc
ADD app.py $HOME/src/app.py
ADD cnn_scripts.py $HOME/src/cnn_scripts.py
ADD config $HOME/src/config
ADD libs $HOME/src/libs
ADD nets $HOME/src/nets
ADD utils $HOME/src/utils
ADD logonic.png $HOME/src/
ADD nic_train_network_batch.py $HOME/src/
ADD nic_infer_segmentation_batch.py $HOME/src/
WORKDIR $HOME/src
